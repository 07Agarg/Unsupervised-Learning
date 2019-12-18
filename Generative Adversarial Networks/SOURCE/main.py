# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:18 2019

@author: ashima.garg
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import data
import model
import config
import tensorflow as tf


if __name__ == "__main__":
    #READ DATA
    train_data = data.DATA()
    train_data.read()
    print("Train Data Loaded")
    #BUILD MODEL
    tf.reset_default_graph()
    model = model.MODEL()
    print("Model Initialized")
    
    #BUILD GENERATOR MODEL
    with tf.variable_scope("generator") as scope:
        gen_out = model.Generator(model.gen_input)
    
    print("Generator Model Built")
        
    #BUILD DISCRIMINATOR MODEL
    with tf.variable_scope("discriminator") as scope:
        model.dis_output1 = model.Discriminator(model.dis_input)
        scope.reuse_variables()
        model.dis_output2 = model.Discriminator(gen_out)
        
    print("Discriminator Model Built")
    
    #TRAIN MODEL
    #gen_vars, dis_vars = model.train(train_data)
    model.train(train_data, gen_out)
    print("Model trained")