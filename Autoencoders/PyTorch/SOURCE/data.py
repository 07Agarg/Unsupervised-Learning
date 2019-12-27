# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:17:33 2019

@author: Ashima
"""

import config
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10


class DATA():
    def __init__(self):
        self.train_dataloader = None
        self.test_loader = None
        self.size = 0

    def read(self, train):
        # Read MNIST Dataset
        if config.DATASET == "MNIST":
            transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5,), (0.5,))])
            dataset = MNIST(root = config.DATA_DIR, train = train, transform = transform, download=True)
            
        # Read CIFAR10 Dataset
        elif config.DATASET == "CIFAR":
            transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = CIFAR10(root = config.DATA_DIR, train = train, transform = transform, download=True)
            
        if train: 
            self.train_dataloader = DataLoader(dataset, config.BATCH_SIZE, shuffle = True)
        else:
            self.test_loader = DataLoader(dataset, config.BATCH_SIZE, shuffle= True)
            
        self.size = len(dataset)
        print(self.size)
        return dataset
    
    def generate_train_batch(self):
        train_iter = iter(self.train_dataloader)
        input_, label_ = next(train_iter)
        #input_ = input_.reshape(input_.size()[0], -1)
        #input_ = input_.view(input_.size(0), -1)
        #print(input_.shape)
        #input_ = (input_ - 0.5)/0.5
        return input_
        