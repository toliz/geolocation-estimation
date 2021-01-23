import logging
import pandas as pd
from math import ceil
from typing import Tuple, Union

import torch
import torchvision
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class GeoDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        ground_truth: bool,
        coords: bool,
        fivecrop = False,
        transform = None,
    ):
        super().__init__()

        self._img_dir = f'datasets/{dataset}/img/'
        self._meta = pd.DataFrame(columns=['fnames', 'cells', 'lat', 'lng'])
        self.fivecrop = fivecrop

        if ground_truth:
            gt = pd.read_csv(f'datasets/{dataset}/meta/gt.csv')
            pnames = gt.columns[1:].tolist()

            self._meta['fnames'] = gt['IMG_ID']
            self._meta['cells'] = gt[pnames].agg(list, axis='columns')

        if coords:
            coords = pd.read_csv(f'datasets/{dataset}/meta/coords.csv')

            if ground_truth:
                coords = coords[coords.IMG_ID.isin(gt.IMG_ID.values)].reset_index()
            else:
                self._meta['fnames'] = coords['IMG_ID']

            self._meta[['lat', 'lng']] = coords[['LAT', 'LON']]

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

    def __getitem__(self, idx):
        fname, cells, lat, lng = self._meta.iloc[idx]

        img = Image.open(f'{self._img_dir}/{fname}')
        img = img.convert("RGB")

        if self.fivecrop: 
            img = torchvision.transforms.Resize(256)(img)
            crops = torchvision.transforms.FiveCrop(224)(img)
            crops_transformed = []
            for crop in crops:
                crops_transformed.append(self._transform(crop))
            img = torch.stack(crops_transformed, dim=0)
        else:
            img = self._transform(img)

        return img, cells, lat, lng

    def __len__(self):
        return len(self._meta)


class GeoDataModule(LightningDataModule):
    def __init__(
        self,
        trainset = 'mp16',
        valset = 'yfcc25k',
        testsets = ['im2gps', 'im2gps3k'],
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
            dataset=self.trainset,
            ground_truth=True,
            coords=False,
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
            dataset=self.valset,
            ground_truth=True,
            coords=True,
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
                dataset=testset,
                ground_truth=False,
                coords=True,
                fivecrop=True,
            )

            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=ceil(self.batch_size/5),
                    num_workers=self.num_workers,
                )
            )
        
        return dataloaders
