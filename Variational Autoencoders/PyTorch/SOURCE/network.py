# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:48:23 2020

@author: Ashima
"""

import config
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder_Linear(nn.Module):
    """Defines multi layer perceptron architecture for autoencoder."""

    def __init__(self, img_size):
        super(VariationalAutoencoder_Linear, self).__init__()
        self.img_size = img_size
        self.encoder = nn.Sequential(
            nn.Linear(img_size, 256),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(p=config.DROPOUT),

            nn.Linear(256, 128),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(p=config.DROPOUT),

            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Dropout(p=config.DROPOUT)
            )
        self.enc_fc1 = nn.Linear(64, config.LATENT_DIM)
        self.enc_fc2 = nn.Linear(64, config.LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(config.LATENT_DIM, 64),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Dropout(p=config.DROPOUT),

            nn.Linear(64, 128),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(p=config.DROPOUT),

            nn.Linear(128, 256),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(p=config.DROPOUT),

            nn.Linear(256, img_size),
            nn.Sigmoid()
            )

    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x, encode=False, decode=False):
        encoded = torch.zeros([], dtype=torch.float32)
        decoded = torch.zeros([], dtype=torch.float32)
        if encode:
            encoded = self.encoder(x.view(-1, self.img_size))
            z_mu = self.enc_fc1(encoded)
            z_var = self.enc_fc2(encoded)
            if decode:
                z = self.reparameterize(z_mu, z_var)
                decoded = self.decoder(z)
            return ((z_mu, z_var), decoded.to(config.device))
        elif decode:
            decoded = self.decoder(x)
        return encoded.to(config.device), decoded.to(config.device)

