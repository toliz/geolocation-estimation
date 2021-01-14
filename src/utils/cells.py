import numpy as np
import pandas as pd

import s2sphere as s2

from pathlib import Path
from typing import Tuple, List


def _hex2bin(hexval):
    thelen = len(hexval) * 4
    binval = bin(int(hexval, 16))[2:]
    while (len(binval)) < thelen:
        binval = "0" + binval

    binval = binval.rstrip("0")
    return binval


def _create_cell(lat, lng, level):
    p1 = s2.LatLng.from_degrees(lat, lng)
    cell = s2.Cell.from_lat_lng(p1)
    cell_parent = cell.id().parent(level)
    hexid = cell_parent.to_token()
    return hexid


class Partitioning:
    def __init__(self, csv_file: str):
        self.name = Path(csv_file).stem
        self._df = pd.read_csv(csv_file)
        self._hexid2class = dict(self._df[['HEX ID', 'CLASS']].values)

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        return f'{self.name} partition with {self.__len__()} classes'

    def get_coords(self, idx: int) -> Tuple[float, float]:
        lat, lng = self._df.iloc[idx][['MEAN LATITUDE', 'MEAN LONGITUDE']]
        return float(lat), float(lng)

    def get_hexid(self, idx: int) -> str:
        return self._df.iloc[idx]['HEX ID']

    def get_class(self, hex_id: str) -> int:
        try:
            return self._hexid2class[hex_id]
        except KeyError as e:
            raise KeyError(f'Unkown hex ID: {hex_id} in {self}')

    def contains(self, idx: int) -> bool:
        return idx in self._hexid2class


class Hierarchy:
    """
        Provides a vector for each partitioning, which maps the
        cells of the partition to the cell of the finest partition.
        
        E.g. if we have 2 partitionings:
        * [1, 2, 3, 4, 5]
        * [1, 2, 3]
        The class will swap them so the last partitioning is the finest.
        Then it will create a matrix that maps the most coarse partition
        to the finest, for example [1, 1, 2, 2, 3], which means that cells
        1 & 2 of the finest partitioning correspond to cell 1 of the coarse.
    """
    
    def __init__(self, partitionings: List[Partitioning]):
        list.sort(partitionings, key=len)

        hierarchy = []
        finest_partitioning = partitionings[-1]

        # Loop through finest partitioning
        for c in range(len(finest_partitioning)):
            cell_bin = _hex2bin(finest_partitioning.get_hexid(c))
            level = int(len(cell_bin[3:-1]) / 2)
            parents = []

            # get parent cells
            for l in reversed(range(2, level + 1)):
                lat, lng = finest_partitioning.get_coords(c)
                hexid_parent = _create_cell(lat, lng, l)
                
                # to coarsest partitioning
                for p in reversed(range(len(partitionings))):
                    if partitionings[p].contains(hexid_parent):
                        parents.append(
                            partitionings[p].get_class(hexid_parent)
                        )

                if len(parents) == len(partitionings):
                    break

            hierarchy.append(parents[::-1])

        self._hierarchy = np.array(hierarchy, dtype=np.int32)

        assert max([len(p) for p in partitionings]) == self._hierarchy.shape[0]
        assert len(partitionings) == self._hierarchy.shape[1]
    
    def __getitem__(self, idx):
        return self._hierarchy[:, idx]
