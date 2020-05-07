# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:10:28 2020

@author: Ashima
"""

import utils
import torch
import config
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.preprocessing import LabelEncoder


class Data():
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
        self.reconstruct_dataloader = DataLoader(dataset, config.TEST_SAMPLES,
                                                 shuffle=True)
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

    def load_features(self, data, net):
        """Extract features from the trained VAE model."""
        net.load_state_dict(torch.load(utils.model_path()))
        net.eval()
        embeddings_list = []
        labels_list = []
        with torch.no_grad():
            for batch in range(int(len(data.dataloader.dataset)/config.BATCH_SIZE)):
                X_batch, Y_batch = data.load_batch()
                (z_mu, z_var), _ = net(X_batch, encode=True, decode=False)
                z_mu = z_mu.cpu().detach().numpy()
                embeddings_list.append(z_mu)
                label_encoder = LabelEncoder()
                labels_list.append(label_encoder.fit_transform(
                    Y_batch.cpu().detach().numpy()))
            embeddings_list = utils.convert_to_list(embeddings_list)
            labels_list = utils.convert_to_list(labels_list)
            return embeddings_list, labels_list
