from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchvision

import torch.nn as nn
import torchmetrics
import torch

import cv2

from torch.utils.data import DataLoader

from phinet import PhiNet


class PNCifar10(pl.LightningModule):
    """Lightning module for benchmarking PhiNets on CIFAR10"""
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        self.model = PhiNet(
            res=32,
            alpha=0.25,
            B0=9,
            beta=1.3,
            t_zero=6,
            squeeze_excite=True, 
            h_swish=True, 
            include_top=True
        )

        self.accuracy = torchmetrics.classification.accuracy.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def loss_fn(self,out,target):
        return nn.CrossEntropyLoss()(out.view(-1, self.num_classes),target)
    
    def configure_optimizers(self):
        LR = 1e-4
        optimizer = torch.optim.AdamW(self.parameters(),lr=LR)
        return optimizer

    def training_step(self,batch,batch_idx):
        x,y = batch
        img = x.view(-1,3,32,32)
        label = y.view(-1)
        out = self(img)
        loss = self.loss_fn(out,label)
        logits = torch.argmax(out,dim=1)
        accu = self.accuracy(logits, label)
        
        self.log('train_acc', accu)
        self.log('train_loss', loss)

        return loss       

    def validation_step(self,batch,batch_idx):
        x,y = batch
        img = x.view(-1,3,32,32)
        label = y.view(-1)
        out = self(img)
        loss = self.loss_fn(out,label)
        logits = torch.argmax(out,dim=1)
        accu = self.accuracy(logits, label)

        self.log('valid_loss', loss)
        self.log('train_acc_step', accu)

        return loss, accu


class Cifar10Dataset(pl.LightningDataModule):
    """Data Module for CIFAR10"""
    def __init__(self, data_path, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.root_set = torchvision.datasets.CIFAR10(data_path, 
                                                train=True, 
                                                download=True, 
                                                transform=transforms.ToTensor())

        self.test_set = torchvision.datasets.CIFAR10(data_path,
                                                train=False, 
                                                download=True, 
                                                transform=transforms.ToTensor())
    
    def setup(self, stage=None):
        self.train_set, self.val_set = train_test_split(self.root_set, test_size = 0.3)

    def train_dataloader(self, stage=None):
        train_loader = DataLoader(self.train_set, batch_size=256, num_workers=16)

        return DataLoader(self.root_set, batch_size=256, num_workers=16)

    # def val_dataloader(self):
    #     valid_loader = DataLoader(self.val_set, batch_size=256, num_workers=16)

    #     return valid_loader

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=256, num_workers=16)
