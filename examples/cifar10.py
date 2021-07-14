from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchvision

import torch.nn as nn
import torchmetrics
import torch

import cv2

from torch.utils.data import DataLoader

from phinet_pl.phinet import PhiNet


class PNCifar10(pl.LightningModule):
    """Lightning module for benchmarking PhiNets on CIFAR10"""
    def __init__(self):
        super(PNCifar10, self).__init__()
        self.classes = 10
        self.lr = 1e-4
        self.model = PhiNet(
            res=32, 
            squeeze_excite=True, 
            h_swish=True, 
            include_top=True
        )

        self.acc = torchmetrics.classification.accuracy.Accuracy()

    def forward(self, x):
        return self.model(x)

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.classes), target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self,batch,batch_idx):
        x, y = batch
        img = x.view(-1, 3, 32, 32)
        label = y.view(-1)

        out = self.forward(img)
        loss = self.loss_fn(out, label)
        logits = torch.argmax(out, dim=1)
        acc = self.acc(logits, label)

        self.log('train_acc', acc)
        self.log('train_loss', loss)

        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch
        img = x.view(-1, 3, 32, 32)
        label = y.view(-1)
        
        out = self.forward(img)
        loss = self.loss_fn(out,label)
        logits = torch.argmax(out, dim=1)
        acc = self.acc(logits, label)

        self.log('valid_loss', loss)
        self.log('train_acc_step', acc)

        return loss, acc


class Cifar10Dataset(pl.LightningDataModule):
    """Dataset module for Cifar10"""
    def __init__(self, data_path: str, batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.root_set = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
    
    def setup(self, stage=None):
        self.train_set, self.val_set = train_test_split(self.root_set, test_size = 0.3)

    def train_dataloader(self, stage=None):
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers=16)

        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=16)
        
        return valid_loader
