# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:48:23 2020

@author: Ashima
"""

import torch
import utils
import config
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
        """Forward computations of the multi-layer autoencoder."""
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


class Encoder(nn.Module):
    """Defines encoder architecture for decoder of CNN based architecture."""

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

    def forward(self, x):
        x = F.dropout(F.leaky_relu(self.bn1(self.conv1(x))),
                      p=config.DROPOUT)
        x = F.dropout(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2),
                      p=config.DROPOUT)
        x = F.dropout(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2),
                      p=config.DROPOUT)
        encoded = x.view(x.size(0),  x.size(1) * x.size(2) * x.size(3))
        # encoded = F.tanh(self.enc(x))
        return encoded.to(config.device)


class Decoder(nn.Module):
    """Defines decoder architecture for decoder of CNN based architecture."""

    def __init__(self):
        super(Decoder, self).__init__()
        self.dec = nn.Linear(in_features=config.LATENT_DIM,
                              out_features=10*10*128)
        self.dbn3 = nn.BatchNorm2d(num_features=128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5,
                                           stride=1, padding=1)
        self.dbn2 = nn.BatchNorm2d(num_features=64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5,
                                          stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(num_features=32)
        self.deconv1 = nn.ConvTranspose2d(32, config.CHANNELS, kernel_size=4,
                                          stride=2, padding=1)

    def forward(self, x):
        z = self.dec(x)
        z = F.dropout(F.leaky_relu(self.dbn3(z.view(z.size(0), 128, 10, 10))),
                      p=config.DROPOUT)
        z = F.dropout(F.leaky_relu(self.dbn2(self.deconv3(z)),
                                   negative_slope=0.2), p=config.DROPOUT)
        z = F.dropout(F.leaky_relu(self.dbn1(self.deconv2(z)),
                                   negative_slope=0.2), p=config.DROPOUT)
        decoded = F.sigmoid(self.deconv1(z))
        return decoded.to(config.device)


class VariationalAutoencoder_CNN(nn.Module):
    """Defines CNN based architecture for autoencoder."""

    def __init__(self):
        super(VariationalAutoencoder_CNN, self).__init__()
        self.encoder = Encoder().to(config.device)
        self.decoder = Decoder().to(config.device)
        self.encoder.apply(utils.weights_init)
        self.decoder.apply(utils.weights_init)
        self.enc_fc1 = nn.Linear(in_features=128*10*10,
                                 out_features=config.LATENT_DIM)
        self.enc_fc2 = nn.Linear(in_features=128*10*10,
                                 out_features=config.LATENT_DIM)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x, encode=False, decode=False):
        """Forward computations of the convolutional autoencoder."""
        encoded = torch.zeros([], dtype=torch.float32)
        decoded = torch.zeros([], dtype=torch.float32)
        if encode:
            encoded = self.encoder(x)
            z_mu = self.enc_fc1(encoded)
            z_var = self.enc_fc2(encoded)
            if decode:
                z = self.reparameterize(z_mu, z_var)
                decoded = self.decoder(z)
            return ((z_mu, z_var), decoded.to(config.device))
        elif decode:
            decoded = self.decoder(x)
        return encoded.to(config.device), decoded.to(config.device)
