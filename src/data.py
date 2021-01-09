from typing import Tuple, Union

import pandas as pd

import torch
import torchvision
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class GeoDataset(Dataset):
    def __init__(self, img_dir: str, ground_truth: str, transform = None):
        super().__init__()

        gt = pd.read_csv(ground_truth)
        pnames = gt.columns[1:].tolist()

        self._img_dir = img_dir
        self._gt = pd.DataFrame({
                'fname': gt.IMG_ID,
                'cells': gt[pnames].agg(list, axis='columns')
        })
        
        if transform == None:
            self._transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self._transform = transform
            self._transform.transforms.extend([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])  

    def __getitem__(self, idx) -> Tuple[torch.tensor, list]:
        fname, cells = self._gt.iloc[idx]
        
        img = Image.open(f'{self._img_dir}/{fname}')
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self._transform(img)
        
        return img, cells

    def __len__(self):
        return len(self._gt)


class GeoDataModule(LightningDataModule):
    def __init__(
        self,
        trainset = 'datasets/mp16',
        valset = 'datasets/yfcc25k',
        testsets = ['datasets/im2gps', 'datasets/im2gps3k'],
        batch_size = 128,
        num_workers = 4,
    ):
        super().__init__()

        self.trainset = trainset
        self.valset = valset
        self.testsets = testsets

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = GeoDataset(
            img_dir=f'{self.trainset}/img',
            ground_truth=f'{self.trainset}/meta/gt.csv',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.66, 1.0)),
            ])
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        return dataloader

    def val_dataloader(self):
        dataset = GeoDataset(
            img_dir=f'{self.valset}/img',
            ground_truth=f'{self.valset}/meta/gt.csv',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
            ])
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        return dataloader

    def test_dataloader(self):
        dataloaders = []

        for testset in self.testsets:
            dataset = GeoDataset(
                img_dir=f'{testset}/img',
                ground_truth=f'{testset}/meta/gt.csv',
            )

            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                )
            )
        
        return dataloaders
