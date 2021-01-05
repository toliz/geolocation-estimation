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


def get_id_s2cell_mapping_from_raw(csv_file) -> pd.DataFrame:
    logging.info('Initialize s2 cells...')

    df = pd.read_csv(csv_file, usecols=['IMG_ID', 'LAT', 'LON'])
    df['IMG_ID'] = df['IMG_ID'].apply(lambda x: x.split('/')[-1])
    df['s2cell'] = df[['LAT', 'LON']].progress_apply(create_s2_cell, axis=1)
    df = df.set_index(df['IMG_ID'])

    return df[['s2cell']]


def assign_class_index(cell: s2.Cell, mapping: dict) -> Union[int, None]:
    for l in range(2, 30):
        cell_parent = cell.id().parent(l)
        hexid = cell_parent.to_token()
        if hexid in mapping:
            return int(mapping[hexid])  # class index

    return None  # valid return since not all regions are covered


def init_logger():
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S',
        level=logging.INFO,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c', '--config',
        type=Path,
        default='models/baseM/config.yml'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = config['model_params']

    for dataset_type in ['val', 'train']:
        dataset_dir = config[f'{dataset_type}_dir']
        label_mapping_file = Path(dataset_dir) / 'meta/meta.csv'
        output_file = Path(dataset_dir) / 'meta/ground_truth.csv'
        output_file.parent.mkdir(exist_ok=True, parents=True)

        logging.info('Load CSV and initialize s2 cells')
        exit()
        df_mapping = get_id_s2cell_mapping_from_raw(label_mapping_file)

        partitioning_files = [Path(p) for p in config['partitionings']['files']]
        for partitioning_file in partitioning_files:
            column_name = partitioning_file.name.split('.')[0]
            logging.info(f'Processing partitioning: {column_name}')
            partitioning = pd.read_csv(
                partitioning_file,
                encoding='utf-8',
                index_col='hex_id',
            )

            # create column with class indexes for respective partitioning
            mapping = partitioning['class_label'].to_dict()
            df_mapping[column_name] = df_mapping['s2cell'].progress_apply(
                lambda cell: assign_class_index(cell, mapping)
            )
            nans = df_mapping[column_name].isna().sum()
            logging.info(
                f'Cannot assign a hexid for {nans} of {len(df_mapping.index)} images '
                f'({nans / len(df_mapping.index) * 100:.2f}%)'
            )

        df_mapping = df_mapping.drop(columns=['s2cell'])  # drop unimportant information
        
        # Remove images that cannot be used
        original_dataset_size = len(df_mapping.index)
        logging.info('Remove all images that could not be assigned a cell')
        df_mapping = df_mapping.dropna()
        logging.info('Remove all images that did not download')
        img_ids = os.listdir(dataset_dir + '/data')
        df_mapping = df_mapping[df_mapping.index.isin(img_ids)]

        column_names = []
        for partitioning_file in partitioning_files:
            column_name = partitioning_file.name.split('.')[0]
            column_names.append(column_name)
            df_mapping[column_name] = df_mapping[column_name].astype('int32')

        df_mapping['targets'] = df_mapping[column_names].agg(list, axis='columns')

        fraction = len(df_mapping.index) / original_dataset_size * 100
        logging.info(
            f'Final dataset size: {len(df_mapping.index)}/{original_dataset_size} ({fraction:.2f})% from original'
        )

        # store final dataset to file
        logging.info(f'Store dataset to {output_file}')

        df_mapping = df_mapping['targets']
        #df_mapping.to_json(output_file, orient='index') # TODO: maybe save to csv for easier reading and uniformity 
        df_mapping.to_csv(output_file)
