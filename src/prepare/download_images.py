import time
import logging
import requests
import PIL
import pandas as pd

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser


def img_resize(img: PIL.Image, size: int) -> PIL.Image:
    """
    Resize an image maintaining the aspect ratio.
    (the smaller edge of the image will be matched to 'size')
    """
    w, h = img.size
    if (w <= size) or (h <= size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), PIL.Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), PIL.Image.BILINEAR)


def flickr_download(image_id, url, size_suffix='z', min_edge_size=None):
    """
    Prevent downloading in full resolution using size_suffix
    For more infor see: https://www.flickr.com/services/api/misc.urls.html
    """
    logger = logging.getLogger('ImageDownloader')

    if size_suffix != '':
        # Modify URL to download image with specific size
        ext = Path(url).suffix
        url = f'{url.split(ext)[0][:-2]}_{size_suffix}{ext}'

    r = requests.get(url)
    if r:
        try:
            image = PIL.Image.open(BytesIO(r.content))
        except PIL.UnidentifiedImageError as e:
            logger.debug(f'{image_id} : {url}: {e}')
            return False
    elif r.status_code == 129:
        time.sleep(60)
        logger.debug('To many requests, sleep for 60s...')
        flickr_download(image_id, url, min_edge_size, size_suffix)
    else:
        logger.debug(f'{image_id} : {url}: {r.status_code}')
        return False

    image = img_resize(image, min_edge_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(args.output.resolve().joinpath(image_id), 'jpeg')
    return True


class ImageDataloader:
    def __init__(self, url_csv: Path, num_images=None):
        logger.info('Read dataset')

        self.df = pd.read_csv(url_csv, names=['image_id', 'url'], header=None, nrows=num_images)
        self.df = self.df.dropna() # remove rows without url
        
        logger.info(f'Number of URLs: {len(self.df.index)}\n')

    def __len__(self):
        return len(self.df.index)

    def __iter__(self):
        for image_id, url in zip(self.df['image_id'].values, self.df['url'].values):
            yield image_id.split('/')[-1], url


def init_logger():
    logger = logging.getLogger('ImageDownloader')
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(str(args.output / 'writer.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def parse_args():
    parser = ArgumentParser(description='Download images for training & validation sets')
    
    parser.add_argument(
        '--url-csv',
        type=Path,
        required=True,
        help='CSV with Flickr image ID and URL for downloading',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory where images are stored',
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Number of images to download')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logger = init_logger()

    imageloader = ImageDataloader(args.url_csv, num_images=args.num_images)

    num_downloaded = 0
    for image_id, url in tqdm(imageloader, unit='img'):
        num_downloaded += flickr_download(image_id, url, size_suffix='z', min_edge_size=320)

    logger.info(f'\nSucesfully downloaded {num_downloaded}/{len(imageloader)} images')
