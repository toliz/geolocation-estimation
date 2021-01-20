import argparse
import logging
import yaml
import os
from pathlib import Path
from typing import Union

import pandas as pd
import s2sphere as s2
from tqdm import tqdm

tqdm.pandas()


def create_s2_cell(latlng):
    p1 = s2.LatLng.from_degrees(latlng['LAT'], latlng['LON'])
    cell = s2.Cell.from_lat_lng(p1)
    return cell


def coords_to_cells(fname: str) -> pd.DataFrame:
    logging.info('Initialize s2 cells...')

    df = pd.read_csv(fname, usecols=['IMG_ID', 'LAT', 'LON'])
    df['s2cell'] = df[['LAT', 'LON']].progress_apply(create_s2_cell, axis=1)

    return df[['IMG_ID', 's2cell']]


def assign_class_index(cell: s2.Cell, mapping: dict) -> Union[int, None]:
    for l in range(2, 30):
        cell_parent = cell.id().parent(l)
        hexid = cell_parent.to_token()
        if hexid in mapping:
            return int(mapping[hexid])  # class index

    return None  # valid return since not all regions are covered


def main(args):
    args.output.parent.mkdir(exist_ok=True, parents=True)

    logging.info('Load CSV and initialize s2 cells')
    df_mapping = coords_to_cells(args.coords)

    for pfile in os.listdir('cells/'):
        pname = pfile.split('.')[0]
        logging.info(f'Processing partitioning: {pname}')
        partitioning = pd.read_csv('cells/' + pfile, encoding='utf-8', index_col='hex_id')

        # Create column with class indexes for the respective partitioning
        mapping = partitioning['class_label'].to_dict()
        df_mapping[pname] = df_mapping['s2cell'].progress_apply(
            lambda cell: assign_class_index(cell, mapping)
        )
        nans = df_mapping[pname].isna().sum()
        logging.info(
            f'Cannot assign a hexid for {nans} of {len(df_mapping.index)} images '
            f'({nans / len(df_mapping.index) * 100:.2f}%)'
        )

    # Drop unimportant information
    df_mapping = df_mapping.drop(columns=['s2cell'])
    
    # Remove images that cannot be used
    original_dataset_size = len(df_mapping.index)
    logging.info('Remove all images that could not be assigned a cell')
    df_mapping = df_mapping.dropna()
    logging.info('Remove all images that did not download')
    img_ids = os.listdir(args.coords.parent.parent / 'img')
    df_mapping = df_mapping[df_mapping['IMG_ID'].isin(img_ids)]

    pnames = []
    for pfile in os.listdir('cells/'):
        pname = pfile.split('.')[0]
        pnames.append(pname)
        df_mapping[pname] = df_mapping[pname].astype('int32')

    fraction = len(df_mapping.index) / original_dataset_size * 100
    logging.info(
        f'Final dataset size: {len(df_mapping.index)}/{original_dataset_size}'
        f'({fraction:.2f})% from original'
    )

    # Store final dataset to file
    logging.info(f'Store dataset to {args.output}')
    df_mapping.to_csv(args.output, index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--coords',
        type=Path,
        required=True,
        help='CSV file with image coordinates',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='CSV file to store cell targets',
    )

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S',
        level=logging.INFO,
    )

    main(args)
