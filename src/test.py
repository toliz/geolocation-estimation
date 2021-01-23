import pickle
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from argparse import ArgumentParser

from data import GeoDataModule
from models import MultiPartitioningClassifier
from pytorch_lightning import Trainer


def main(args):
    # Init model & data module
    model = MultiPartitioningClassifier(name=args.model, reset_classifier=False)
    datamodule = GeoDataModule()
    
    # Init Trainer
    trainer = Trainer(gpus=args.gpus, precision=args.precision)
    
    # Train & validate
    print()
    trainer.test(model, datamodule=datamodule, verbose=False)
    print()

    results = pickle.load(open(f'models/{args.model}/test_report.pkl', 'rb'))

    # Formatting results
    for result, name in zip(results, datamodule.testsets):
        df = pd.DataFrame(result).T
        df.index.name = name
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt='.2f'))
        print()


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
    