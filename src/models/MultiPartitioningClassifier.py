import utils
import pickle
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule


class MultiPartitioningClassifier(LightningModule):
    def __init__(
        self,
        name = 'baseM',
        architecture = 'resnet50',
        cell_dir = 'cells/',
        load_hierarchy = True,
        load_pretrained = False,
    ):
        super().__init__()

        if type(cell_dir) is not Path:
            cell_dir = Path(cell_dir)

        # Build hierarchy
        self.partitionings = [utils.cells.Partitioning(pfile) for pfile in cell_dir.iterdir()]
        if load_hierarchy:
            self.hierarchy = pickle.load(open(f'models/{name}/hierarchy.pkl', 'rb'))
        else:
            self.hierarchy = utils.cells.Hierarchy(self.partitionings)
            pickle.save(self.hierarchy, open(f'models/{name}/hierarchy.pkl', 'wb'))

        # Build backbone network
        backbone = torchvision.models.__dict__[architecture](pretrained=True)
        
        if 'resnet' in architecture:
            nfeatures = backbone.fc.in_features
            self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise NotImplementedError
        
        self.backbone.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.backbone.flatten = torch.nn.Flatten(start_dim=1)

        # Build classifiers
        self.classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(nfeatures, len(partitioning)) for partitioning in self.partitionings]
        )

        # Load weights
        if load_pretrained:
            self.load('models/{name}/pretrained.ckpt')

    def load(self, pretrained_path: str):
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)

        state_dict_features = OrderedDict()
        state_dict_classifier = OrderedDict()
        for k, w in checkpoint['state_dict'].items():
            if k.startswith('model'):
                state_dict_features[k.replace('model.', '')] = w
            elif k.startswith('classifier'):
                state_dict_classifier[k.replace('classifier.', '')] = w
            else:
                logging.warning(f'Unexpected prefix in state_dict: {k}')
        self.backbone.load_state_dict(state_dict_features, strict=True)
        self.classifiers.load_state_dict(state_dict_classifier, strict=True)


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
