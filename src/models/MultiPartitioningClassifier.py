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

        # Load pretrained
        """if self.hparams.pretrained:
            logger.info('Load weights from pre-trained model')
            load_weights(self, self.hparams.pretrained)"""

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

        """# log top-k accuracy for each partitioning
        individual_accuracy_dict = utils_global.accuracy(
            output, target, [p.shortname for p in self.partitionings]
        )
        
        # log loss for each partitioning
        individual_loss_dict = {
            f'loss_val/{p}': l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }

        # log GCD error@km threshold
        distances_dict = {}

        if self.hierarchy is not None:
            hierarchy_logits = [
                yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(output)
            ]
            hierarchy_logits = torch.stack(hierarchy_logits, dim=-1,)
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        pnames = [p.shortname for p in self.partitionings]
        if self.hierarchy is not None:
            pnames.append('hierarchy')
        for i, pname in enumerate(pnames):
            # get predicted coordinates
            if i == len(self.partitionings):
                i = i - 1
                pred_class_indexes = torch.argmax(hierarchy_preds, dim=1)
            else:
                pred_class_indexes = torch.argmax(output[i], dim=1)
            pred_latlngs = [
                self.partitionings[i].get_lat_lng(idx)
                for idx in pred_class_indexes.tolist()
            ]
            pred_lats, pred_lngs = map(list, zip(*pred_latlngs))
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            # calculate error
            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                true_lats.type_as(pred_lats),
                true_lngs.type_as(pred_lats),
            )
            distances_dict[f'gcd_{pname}_val'] = distances

        output = {
            'loss_val/total': loss,
            **individual_accuracy_dict,
            **individual_loss_dict,
            **distances_dict,
        }
        return output"""
        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.01)#, **self.hparams.optim['params'])
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.5, milestones=[4, 8, 12, 13, 14, 15])#, **self.hparams.scheduler['params'])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
