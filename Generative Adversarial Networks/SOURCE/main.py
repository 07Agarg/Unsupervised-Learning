# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:18 2019

@author: ashima.garg
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = 1
import data
import model
import config


if __name__ == "__main__":
    #READ DATA
    train_data = data.DATA(config.TRAIN_DIR)
    print("Train Data Loaded")
    #BUILD MODEL
    model = model.MODEL()
    print("Model Initialized")
    model.build()
    print("Model Built")
    #TRAIN MODEL
    model.train(train_data)
    print("Model trained")
    #TEST MODEL
    #READ TEST DATA
    test_data = data.DATA(config.TEST_DIR)
    print("Test Data Loaded")
    model.test(test_data)
    print("Model Tested")