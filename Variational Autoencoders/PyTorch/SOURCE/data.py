# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:10:28 2020

@author: Ashima
"""

import config
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


class DATA():
    def __init__(self):
        self.dataloader = None
        self.iterator = None

    def read(self, train=True):
        # Read MNIST Dataset
        """Read dataset."""
        if config.DATASET == "MNIST":
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = MNIST(root=config.DATA_DIR, train=train,
                            transform=transform, download=True)
        else:
            raise ValueError('Other datasets are not supported!')
        self.dataloader = DataLoader(dataset, config.BATCH_SIZE, shuffle=True)
        self.iterator = iter(self.dataloader)
        self.dataset = dataset
        return dataset

    def load_batch(self):
        """Load batch of data."""
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            x, y = next(self.iterator)
        return [torch.Tensor(x.float()).to(config.device),
                torch.Tensor(y.float()).to(config.device)]
