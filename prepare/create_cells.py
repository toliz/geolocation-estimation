import csv
import os
import logging
import sys
import argparse
from time import time
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from collections import Counter

import pandas as pd
import s2sphere as s2


""" S2 Utils """

def create_s2_cell(lat, lng):
    p1 = s2.LatLng.from_degrees(lat, lng)
    cell = s2.Cell.from_lat_lng(p1)
    return cell

def create_cell_at_level(cell, level):
    cell_parent = cell.id().parent(level)
    hex_id = cell_parent.to_token()
    return hex_id

def s2_info(img, level):
    cell = create_s2_cell(img[1], img[2])
    hex_id = create_cell_at_level(cell, level)
    return [*img, hex_id, cell]


""" Cell Utils """

def init_cells(img_container_0, level, num_threads):
    logging.info('Initialize cells of level {} ...'.format(level))
    
    init = time()
    f = partial(s2_info, level=level)
    img_container = []
    with Pool(num_threads) as p:
        for x in p.imap_unordered(f, img_container_0, chunksize=1000):
            img_container.append(x)
    logging.debug(f'Time multiprocessing: {time() - init:.2f}s')

    start = time()
    h = dict(Counter(list(list(zip(*img_container))[3])))       # counts how many times each cell
    logging.debug(f'Time creating h: {time() - start:.2f}s')    # has been assigned
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}\n')

    return img_container, h

def delete_cells(img_container, h, t_min):
    logging.info(f'Remove cells with |img| < {t_min} ...')
    start = time()

    del_cells = {k for k, v in h.items() if v <= t_min}
    h = {k: v for k, v in h.items() if v > t_min}

    img_container_f = []
    for img in img_container:
        hex_id = img[3]
        if hex_id not in del_cells:
            img_container_f.append(img)
    
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}\n')

    return img_container_f, h

def gen_subcells(img_container_0, h_0, level, t_max):
    logging.info('Level {}'.format(level))
    start = time()

    img_container = []
    h = {}
    for img in img_container_0:
        hex_id_0 = img[3]
        if h_0[hex_id_0] > t_max:
            hex_id = create_cell_at_level(img[4], level)
            img[3] = hex_id
            try:
                h[hex_id] = h[hex_id] + 1
            except:
                h[hex_id] = 1
        else:
            try:
                h[hex_id_0] = h[hex_id_0] + 1
            except:
                h[hex_id_0] = 1
        img_container.append(img)

    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}\n')

    return img_container, h


""" Main utils """

def write_output(fname, img_container, h):
    with open(fname, mode='w') as f:
        logging.info(f'Write to {f.name}')

        cells_writer = csv.writer(f, delimiter=',')
        # write column names
        cells_writer.writerow(
            [
                'CLASS',
                'HEX ID',
                'IMAGES PER CELL',
                'MEAN LATITUDE',
                'MEAN LONGITUDE',
            ]
        )

        # write dict
        i = 0
        cell2class = {}
        coords_sum = {}

        # generate class ids for each hex cell id
        for k in h.keys():
            cell2class[k] = i
            coords_sum[k] = [0, 0]
            i = i + 1

        # calculate mean GPS coordinate in each cell
        for img in img_container:
            coords_sum[img[3]][0] = coords_sum[img[3]][0] + img[1]
            coords_sum[img[3]][1] = coords_sum[img[3]][1] + img[2]

        # write partitioning information
        for k, v in h.items():
            cells_writer.writerow(
                [cell2class[k], k, v, coords_sum[k][0] / v, coords_sum[k][1] / v]
            )


def main(args):
    # Read dataset
    df = pd.read_csv(args.dataset, usecols=['IMG_ID', 'LAT', 'LON'])
    img_container = list(df.itertuples(index=False, name=None))
    num_images = len(img_container)
    logging.info('{} images available.\n'.format(num_images))
    level = args.lvl_min

    # Initialize cells
    img_container, h = init_cells(img_container, level, args.threads)    
    img_container, h = delete_cells(img_container, h, args.img_min)

    # Recursively split cells with > t_max images
    logging.info('Create subcells ...')
    while any(v > args.img_max for v in h.values()) and level < args.lvl_max:
        level = level + 1
        img_container, h = gen_subcells(img_container, h, level, args.img_max)

    # Remove cells with < t_min images
    img_container, h = delete_cells(img_container, h, args.img_min)
    logging.info(f'Final number of images: {len(img_container)}')

    # Save partitioning
    logging.info('Write output file ...')
    write_output(f'cells/{args.pname}.csv', img_container, h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Cell Partitioning')
    
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Verbose output',
    )
    parser.add_argument(
        '--pname',
        type=str,
        required=True,
        help='Partitioning name',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=Path('datasets/mp16/meta/coords.csv'),
        help='Path to dataset csv file',
    )
    parser.add_argument(
        '--img-min',
        type=int,
        required=True,
        help='Minimum number of images per geographical cell',
    )
    parser.add_argument(
        '--img-max',
        type=int,
        required=True,
        help='Maximum number of images per geographical cell',
    )
    parser.add_argument(
        '--lvl-min',
        type=int,
        required=False,
        default=2,
        help='Minimum partitioning level',
    )
    parser.add_argument(
        '--lvl-max',
        type=int,
        required=False,
        default=30,
        help='Maximum partitioning level',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of threads to download and process images',
    )

    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S',
        level=level,
    )

    main(args)