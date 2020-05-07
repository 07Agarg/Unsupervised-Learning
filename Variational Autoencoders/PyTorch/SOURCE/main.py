# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:53:51 2020

@author: Ashima
"""

import os
import data
import model
import torch
import utils
import config
import network
import model_svm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    # READ DATA
    train_data = data.Data()
    train_data.read(train=True)   # True for train data.
    print("Loading " + config.DATASET + " train dataset...")

    test_data = data.Data()
    test_data.read(train=False)  # False for test data.
    print("Loading " + config.DATASET + " test dataset...")

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
    print(net)

    modeloperator = model.Operators(net)
    print("Model Built")

    modeloperator.train(train_data, test_data)
    print("Model Trained")

    utils.plot_latent_tsne(train_data, net, plot_type="TRAIN")
    print("Plotted train dataset tSNE of Latent Space")

    utils.plot_latent_tsne(test_data, net, plot_type="TEST")
    print("Plotting test dataset tSNE of Latent Space")

    model = model_svm.Model()
    model.svc_train(train_data, net)
    print("SVM Model Trained")

    model.svc_test(test_data, net)
    print("SVM Model Tested")
