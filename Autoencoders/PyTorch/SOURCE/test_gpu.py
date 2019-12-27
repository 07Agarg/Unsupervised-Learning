# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:17:33 2019

@author: Ashima
"""

# import torch
# print(torch.rand(2, 3).cuda())
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

import os
import config
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST(config.DATA_DIR, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)