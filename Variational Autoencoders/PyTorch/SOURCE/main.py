# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:53:51 2020

@author: Ashima
"""

import os
import config
import data
import model
import network
import torch
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    # READ DATA
    data = data.DATA()
    data.read(train=True)   # True for train data.
    print("Loading " + config.DATASET + "...")

    if config.DATASET == "MNIST":
        config.IMG_DIM = 28
        config.CHANNELS = 1
    config.IMAGE_SIZE = config.CHANNELS * config.IMG_DIM * config.IMG_DIM

    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("Using CPU")

    # BUILD MODEL
    if config.NETWORK_TYPE == "LINEAR":
        net = network.VariationalAutoencoder_Linear(config.IMAGE_SIZE).to(
            config.device)
    elif config.NETWORK_TYPE == "CNN":
        net = network.VariationalAutoencoder_CNN().to(config.device)
    print("Model Initialized")

    modeloperator = model.Operators(net, data)
    print("Model Built")

    modeloperator.train(data)
    print("Model Trained")

    data.read(train=False)  # False for test data.
    print("Test Data Loaded")

    utils.plot_latent_tsne(data, net)
    print("Plotted TSNE of Latent Space")
