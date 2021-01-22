import sys
import logging
import collections
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


class CustomDict(collections.UserDict):
    """
    An entry of the dictionary can be accessed if the query key contains the
    entry key as a substring delimited by '_'.
    """

    def __missing__(self, key):
        init_key = key
        key = str(key)
        if key.endswith('.jpg'):
            key = key[:-4]
        split = key.split('_', 1)
        if len(split) > 1:
            if split[0] in self.data:
                return self[split[0]]
            else:
                return self[split[1]]
        raise KeyError(init_key)

    def __contains__(self, key):
        key = str(key)
        if key in self.data:
            return True
        else:
            split = key.split('_', 1)
            if len(split) > 1:
                if split[0] in self.data:
                    return True
                else:
                    return split[1] in self
            return False


def extract_tags(text: str, prob_threshold=0.9):
    tags = []
    
    for item in text.split(','):
        tag, prob = item.split(':')

        if float(prob) > prob_threshold:
            tags.append(tag)

    return tags


def main(args):
    # Parse tags in dictionary
    filesize = Path(args.autotag_file).stat().st_size
    tag_dict = CustomDict()

    logging.info('Reading autotag file...')
    pbar = tqdm(total=filesize, file=sys.stdout)
    with open(args.autotag_file, 'r') as f:
        for line in f:
            id, tags = line.split('\t')
            
            # Skip images with no tags provided
            if tags == '\n':
                pbar.update(len(line))
                continue

            tags = extract_tags(tags) 
            if tags:
                tag_dict[id] = tags

            pbar.update(len(line))
    pbar.close()

    # Apply dictionary to dataset images
    logging.info(f'Applying to {args.dataset} images...')
    df = pd.read_csv(f'datasets/{args.dataset}/meta/gt.csv')

    def tags(id):
        try:
            return tag_dict[id]
        except KeyError:
            return float('nan')

    df['TAGS'] = df['IMG_ID'].progress_apply(tags)
    df[['IMG_ID', 'TAGS']].to_csv(f'datasets/{args.dataset}/meta/autotags.csv', index=False)

    if args.dataset == 'mp16':
        # Update cell files
        for pfile in Path(args.cell_dir).iterdir():
            pname = pfile.stem
            logging.info(f'Grouping by {pname} partition...')

            # Group tags by cell
            ptags = df[[pname, 'TAGS']].dropna()
            ptags = ptags.groupby(pname).agg(sum)
            ptags['TAGS'] = ptags['TAGS'].apply(lambda x: list(set(x)))

            # Add tags to current cell csv
            cells = pd.read_csv(pfile, index_col='CLASS')
            cells['TAGS'] = ptags
            cells.to_csv(pfile, float_format='%.6f')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--autotag-file',
        type=str,
        help='File containing YFCC100M autotags'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mp16',
        choices=['mp16', 'yfcc25k', 'im2gps3k', 'im2gps'],
    )
    parser.add_argument(
        '--cell-dir',
        type=str,
        default='cells/',
        help='Partitioning dir.'
    )
    args = parser.parse_args()

    logging.basicConfig(
        format='\n%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S',
        level=logging.INFO
    )

    tqdm.pandas()

    main(args)
