from argparse import Namespace
import logging

import torch
import torch.nn.functional as F
import torchvision

from pytorch_lightning import LightningModule
from utils.hierarchy import Partitioning, Hierarchy


class MultiPartitioningClassifier(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Build hierarchy
        logging.info("Building hierarchy...")
        self.partitionings = [Partitioning(f'cells/{pfile}.csv') for pfile in hparams['partitionings']]
        #self.hierarchy = Hierarchy(self.partitionings)

        # Build backbone network
        logging.info("Building backbone network...")
        backbone = torchvision.models.__dict__[hparams['arch']](pretrained=True)
        
        if "resnet" in hparams['arch']:
            out_features = backbone.fc.in_features
            self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise NotImplementedError
        
        self.backbone.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.backbone.flatten = torch.nn.Flatten(start_dim=1)

        # Build classifiers
        logging.info("Building classifiers...")
        self.classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(out_features, len(partitioning)) for partitioning in self.partitionings]
        )

    def forward(self, x):
        features = self.backbone(x)
        preds = [classifier(features) for classifier in self.classifiers]

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.__dict__[self.hparams['optimizer']['name']](
            self.parameters(),
            **self.hparams['optimizer']['params']
        )
        
        scheduler = torch.optim.lr_scheduler.__dict__[self.hparams['scheduler']['name']](
            optimizer,
            **self.hparams['scheduler']['params']
        )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        losses = [F.cross_entropy(pred, target) for pred, target in zip(preds, targets)]
        loss = sum(losses)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        losses = [F.cross_entropy(pred, target) for pred, target in zip(preds, targets)]
        loss = sum(losses)

        return loss
