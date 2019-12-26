# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:17:33 2019

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
    data = data.DATA()
    data.read(True)   # True for train data
    print("Train Data Loaded")
    
    if torch.cuda.is_available():
        print("Using CUDA")    
        #BUILD MODEL
        net = model.Autoencoder().cuda()
        print("Model Initialized")
        
        modeloperator = model.Operators(net)
        print("Model Built")
        
        modeloperator.train(data)
        print("Model Trained")
        
    
        
     