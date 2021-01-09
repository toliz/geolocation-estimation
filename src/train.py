from argparse import ArgumentParser
from pathlib import Path

from models import MultiPartitioningClassifier
from data import GeoDataModule
from pytorch_lightning import Trainer


def parse_args():
    args = ArgumentParser()

    args.add_argument("-c", "--config", type=Path, default=Path("config/baseM.yml"))
    args.add_argument("--progbar", action="store_true")
    
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Init model & data module
    model = MultiPartitioningClassifier({})
    datamodule = GeoDataModule()

    # Init Trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=15,
        precision=16,
        progress_bar_refresh_rate=1,
    )

    # Train & validate
    trainer.fit(model, datamodule)
