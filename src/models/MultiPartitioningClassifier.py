from argparse import Namespace

import torch
import torchvision
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler

from pytorch_lightning import LightningModule

class MultiPartitioningClassifier(LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

        self.__build_model('resnet50')

    def __build_model(self, arch):
        # Build backbone network
        backbone = torchvision.models.__dict__[arch](pretrained=True)
        
        if "resnet" in arch:
            # Usually all ResNet variants
            nfeatures = backbone.fc.in_features
            self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise NotImplementedError
        
        self.backbone.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.backbone.flatten = torch.nn.Flatten(start_dim=1)

        # Build classifiers
        self.classifiers = torch.nn.ModuleList(
            #[torch.nn.Linear(nfeatures, len(partitioning)) for partitioning in self.partitionings]
            [
                torch.nn.Linear(nfeatures, 3439),
                torch.nn.Linear(nfeatures, 7561),
                torch.nn.Linear(nfeatures, 13662),
            ]
        )

    def forward(self, x):
        features = self.backbone(x)
        preds = [classifier(features) for classifier in self.classifiers]

        return preds

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

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.01
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.5, milestones=[4, 8, 12, 13, 14, 15])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
