# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:17:33 2019

@author: Ashima
"""

import os
import config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import data
import model
import network
import torch

if __name__ == "__main__":
    #READ DATA
    data = data.DATA()
    data.read(True)   # True for train data
    print("Train Data Loaded")
    
    if config.DATASET == "MNIST":
        config.IMG_DIM = 28
        config.CHANNELS = 1
    elif config.DATASET == "CIFAR":
        config.IMG_DIM = 32
        config.CHANNELS = 3
        
    config.IMAGE_SIZE = config.CHANNELS*config.IMG_DIM*config.IMG_DIM
        
    if torch.cuda.is_available():
        print("Using CUDA")    
        
        #BUILD MODEL
        #net = network.Autoencoder(config.IMAGE_SIZE).cuda()
        net = network.Autoencoder_CNN(config.IMAGE_SIZE, config.CHANNELS).cuda()
        print("Model Initialized")
        
        modeloperator = model.Operators(net)
        print("Model Built")
        
        modeloperator.train(data)
        print("Model Trained") 
        
    
        
     