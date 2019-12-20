# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:55:58 2019

@author: Ashima
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import data
import model
import utils
import config
import torch

if __name__ == "__main__":
    #READ DATA
    train_data = data.DATA()
    train_data.read()
    print("Train Data Loaded")
    #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    if torch.cuda.is_available():
        print("Using CUDA")    
        #BUILD MODEL
        generator_net = model.Generator_Model().cuda()
        generator_net.apply(utils.weights_init)
        print("Generator Model Initialized")
        
        discriminator_net = model.Discriminator_Model().cuda()
        discriminator_net.apply(utils.weights_init)
        print("Discriminator Model Initializes")
        
        modeloperator = model.Operators(generator_net, discriminator_net)
        print("Model Built")
        
        modeloperator.train(train_data)
        print("Model Trained")
     