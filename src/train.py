from argparse import ArgumentParser
from pathlib import Path
import yaml
import logging

from models import MultiPartitioningClassifier
from data import GeoDataModule
from pytorch_lightning import Trainer


def parse_args():
    args = ArgumentParser()

    args.add_argument(
        "-m", "--model-dir",
        type=Path,
        default=Path("models/baseM"),
        help="Path to model folder",
    )
    
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S',
        level=logging.INFO,
    )

    with open(args.model_dir / 'config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        model_params = config["model_params"]
        trainer_params = config["trainer_params"]

    # Init model & data module
    model = MultiPartitioningClassifier(model_params)
    datamodule = GeoDataModule()

    # Init Trainer
    trainer = Trainer(
        **trainer_params,
        progress_bar_refresh_rate=1,
    )

    # Train & validate
    trainer.fit(model, datamodule)
