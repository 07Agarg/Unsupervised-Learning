# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:55:58 2019

@author: Ashima
"""

import config
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

class DATA():
    def __init__(self):
        self.dataset = None
        self.train_dataloader = None
        self.size = 0

    def read(self):
        trans = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        print(config.DATA_DIR)
        self.dataset = MNIST(root = config.DATA_DIR, train = True, transform = trans, download=True)
        self.train_dataloader = DataLoader(self.dataset, config.BATCH_SIZE, shuffle = True)
        self.size = len(self.dataset)
        print(self.size)
        return self.dataset
    
    def generate_train_batch(self):
        train_iter = iter(self.train_dataloader)
        input_, label_ = next(train_iter)
        input_ = input_.reshape(input_.size()[0], -1)
        input_ = (input_ - 0.5)/0.5
        return input_, label_
        