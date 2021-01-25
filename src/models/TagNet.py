import pandas as pd
from pathlib import Path

class TagNet():
    def __init__(self, cell_dir='cells/'):
        cell_dir = Path(cell_dir)

        self.pnames = []
        self.partitionings = []

        for pname in cell_dir.iterdir():
            partitioning = pd.read_csv(pname)

            self.pnames.append(pname.stem)
            self.partitionings.append(partitioning)

    def __call__(self, fname):
        tags = self.tags[fname] # This will be extracted by the image eventually

        preds = {
            pname: partitioning['TAGS'].apply(lambda x: sum(el in tags for el in x) / len(tags))
            for (pname, partitioning) in zip(self.pnames, self.partitionings)
        }
