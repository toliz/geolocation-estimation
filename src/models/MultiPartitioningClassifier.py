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
        self.name = name

        if type(cell_dir) is not Path:
            cell_dir = Path(cell_dir)

        # Build hierarchy
        self.partitionings = [utils.cells.Partitioning(pfile) for pfile in cell_dir.iterdir()]

        if load_hierarchy:
            list.sort(self.partitionings, key=len)
            self.hierarchy = pickle.load(open(f'models/{name}/hierarchy.pkl', 'rb'))
        else:
            self.hierarchy = utils.cells.Hierarchy(self.partitionings)
            pickle.dump(self.hierarchy, open(f'models/{name}/hierarchy.pkl', 'wb'))

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
            self.load_state_dict(torch.load('models/baseM/weights.pt'))

    def forward(self, images):
        assert isinstance(images, torch.Tensor)
        assert (images.ndim == 5) # batch, crops, channels, height, width
        
        # Reshape crop dimension to batch
        batch_size, num_crops = images.shape[:2]
        images = torch.reshape(images, (batch_size * num_crops, *images.shape[2:]))

        # Forward pass
        features = self.backbone(images)
        preds = [classifier(features) for classifier in self.classifiers]
        preds = [F.softmax(pred, dim=1) for pred in preds]

        # Respape back to access individual crops
        preds = [
            torch.reshape(pred, (batch_size, num_crops, *list(pred.shape[1:]))) for pred in preds
        ]

        # Calculate max over crops
        preds = [torch.max(pred, dim=1)[0] for pred in preds]

        # Hierarchical prediction
        hierarchy_logits = torch.stack(
            [pred[:, self.hierarchy[i]] for i, pred in enumerate(preds)],
            dim=-1,
        )
        hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        # Get latitude & longitute
        for i, prob in enumerate(preds):
            preds[i] = torch.argmax(prob, dim=1)
            preds[i] = [self.partitionings[i].get_coords(c.item()) for c in preds[i]]

        hierarchy_preds = torch.argmax(prob, dim=1)
        hierarchy_preds = [self.partitionings[i].get_coords(c.item()) for c in hierarchy_preds]
        
        preds = dict(zip([p.name for p in self.partitionings], preds))
        preds['hierarchy'] = hierarchy_preds

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001,
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            gamma=0.5,
            milestones=[4, 8, 12, 13, 14, 15],
        )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        images, targets, _, _ = batch

        # Forward pass
        features = self.backbone(images)
        preds = [classifier(features) for classifier in self.classifiers]

        # Loss
        losses = [F.cross_entropy(pred, target) for pred, target in zip(preds, targets)]
        loss = sum(losses)

        # Logging
        self.log(f'train_loss', loss, logger=True)
        for p, l in zip(self.partitionings, losses):
            self.log(f'train_loss\{p.name}', l, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, _, _ = batch

        # Forward pass
        features = self.backbone(images)
        preds = [classifier(features) for classifier in self.classifiers]

        # Loss
        losses = [F.cross_entropy(pred, target) for pred, target in zip(preds, targets)]
        loss = sum(losses)

        # Logging
        self.log(f'val_loss', loss, logger=True)
        for p, l in zip(self.partitionings, losses):
            self.log(f'val_loss\{p.name}', l, logger=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx = None):
        images, _, true_lat, true_lng = batch
        pred_coords = self(images)

        GCD = {}
        for pname, pred_coord in pred_coords.items():
            pred_lat, pred_lng = zip(*pred_coord)

            pred_lat = torch.tensor(pred_lat, dtype=torch.float, device=true_lat.device)
            pred_lng = torch.tensor(pred_lng, dtype=torch.float, device=true_lng.device)

            GCD[pname] = utils.report.GCD(pred_lat, pred_lng, true_lat, true_lng)

        return GCD

    def test_epoch_end(self, outputs):
        results = []

        if not isinstance(outputs[0], list):
            outputs = [outputs]

        pnames = [partitioning.name for partitioning in self.partitionings]
        for output in outputs:
            accuracies = {}

            for pname in pnames:
                GCD = torch.cat([o[pname] for o in output], dim=0)
                accuracies[pname] = utils.report.acc(GCD)

            results.append(accuracies)
        
        pickle.dump(results, open(f'models/{self.name}/test_report.pkl', 'wb'))
