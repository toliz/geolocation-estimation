from argparse import ArgumentParser
from pathlib import Path

from data import GeoDataModule
from models import MultiPartitioningClassifier
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def main(args):
    # Init model & data module
    model = MultiPartitioningClassifier(name=args.model, cell_dir='cells-mine/')
    datamodule = GeoDataModule()

    # Init Trainer
    tb_logger = TensorBoardLogger(f'models/{args.model}', name='tb_logs')
    trainer = Trainer(gpus=args.gpus, precision=args.precision, logger=tb_logger)

    # Train & validate
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '-m', '--model',
        type=str,
        default='baseM',
        help='Model name. Your model will be placed in <project-folder>/models/<model>',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of gpus to train on (int) or which GPUs to train on (list or str) per node',
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=16,
        help='32 for full precision or 16 for half precision. Available for CPUs, GPUs or TPUs',
    )

    args = parser.parse_args()

    main(args)
