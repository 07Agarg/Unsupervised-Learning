# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:13:17 2019

@author: Ashima
"""

import torch 
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, img_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size, 256), 
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(), 
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(), 
            nn.Linear(256, img_size),
            nn.Tanh(),
            )
        #print("img_size: ", img_size)
        
    def forward(self, x, train):
        if train:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        else:
            decoded = self.decoder(x)
        return decoded.cuda()
    
class Autoencoder_CNN(nn.Module):
    def __init__(self, img_size, channels):
        super(Autoencoder_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 12, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, 2, 1), 
            nn.ReLU(),
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, 4, 2, 1), 
            nn.ReLU(), 
            nn.ConvTranspose2d(12, 3, 4, 2, 1), 
            nn.Tanh(),
            )
        
    def forward(self, x, train):
        if train:
            encoded = self.encoder(x)
            #print(encoded.type)
            #print("encoded.size: ", encoded.size())
            decoded = self.decoder(encoded)
        else:
            decoded = self.decoder(x)
        return decoded.cuda()