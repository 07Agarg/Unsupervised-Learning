# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:20 2019

@author: ashima.garg
"""

import os
import config
import itertools
import numpy as np
import neural_network
import tensorflow as tf
import matplotlib.pyplot as plt

class MODEL():
    
    def __init__(self):
        self.gen_input = tf.placeholder(shape = [None, 100], dtype = tf.float32)
        self.dis_input = tf.placeholder(shape = [None, config.IMAGE_SIZE], dtype = tf.float32) 
        self.dis_output1 = None            #D(x)
        self.dis_output2 = None            #D(G(z))
        self.gen_loss = None
        self.dis_loss = None
        self.save_path = None
        self.gen_losses = []
        self.dis_losses = []
        
    def Generator(self, Z):
        dim = Z.get_shape()[1].value
        hidden1 = neural_network.Generator_Layer(shape = [dim, 256], name = "gn_1", stddev = 0.02, value = 0.1)
        h1 = hidden1.feed_forward_lrelu(Z)
        
        hidden2 = neural_network.Generator_Layer(shape = [256, 512], name = "gn_2", stddev = 0.02, value = 0.1)
        h2 = hidden2.feed_forward_lrelu(h1)
        
        hidden3 = neural_network.Generator_Layer(shape = [512, 1024], name = "gn_3", stddev = 0.02, value = 0.1)
        h3 = hidden3.feed_forward_lrelu(h2)
        
        hidden4 = neural_network.Generator_Layer(shape = [1024, 784], name = "gn_4", stddev = 0.02, value = 0.1)
        h4 = hidden4.feed_forward_tanh(h3)
        
        return h4
    
    def Discriminator(self, x):
        dim = x.get_shape()[1].value
        hidden1 = neural_network.Discriminator_Layer(shape = [dim, 1024], name = "dis_1", stddev = 0.01, value = 0.1)
        h1 = hidden1.feed_forward_1(x)
        
        hidden2 = neural_network.Discriminator_Layer(shape = [1024, 512], name = "dis_2", stddev = 0.01, value = 0.1)
        h2 = hidden2.feed_forward_1(h1)
        
        hidden3 = neural_network.Discriminator_Layer(shape = [512, 256], name = "dis_3", stddev = 0.01, value = 0.1)
        h3 = hidden3.feed_forward_1(h2)
        
        hidden4 = neural_network.Discriminator_Layer(shape = [256, 1], name = "dis_4", stddev = 0.01, value = 0.1)
        h4 = hidden4.feed_forward_2(h3)
        
        return h4  
    
    def loss_fun(self):
        eps = 1e-2
        self.dis_loss = tf.reduce_mean(-tf.log(self.dis_output1 + eps) - tf.log(1. - self.dis_output2+eps))
        self.gen_loss = tf.reduce_mean(-tf.log(self.dis_output2 + eps))
    
    def train(self, data, gen_out):
        self.loss_fun()
        t_vars = tf.trainable_variables()
        gen_vars = [var for var in t_vars if "generator" in var.name]
        dis_vars = [var for var in t_vars if "discriminator" in var.name]
        
        #return gen_vars, dis_vars
        
        gen_optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.gen_loss, var_list = gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.dis_loss, var_list = dis_vars)
        
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
            init = tf.global_variables_initializer()
            session.run(init)
            print("Variables initialized.... ")
            print(tf.test.gpu_device_name())
            print(tf.test.is_built_with_cuda())
            print(tf.test.is_gpu_available())            
            for epoch in range(config.NUM_EPOCHS):
                G_losses = []
                D_losses = []
                for batch in range(int(data.size/config.BATCH_SIZE)):
                    batchX, batchY = data.generate_train_batch()
                    z = np.random.uniform(-1, 1, (config.BATCH_SIZE, 100))
                    #discriminator step
                    feed_dict = {self.dis_input: batchX, self.gen_input: z}
                    dis_loss_, _ = session.run([self.dis_loss, dis_optimizer], feed_dict = feed_dict)
                    D_losses.append(dis_loss_)
                    
                    #generator step
                    #z = np.random.normal(0, 1, (config.BATCH_SIZE, 100))
                    feed_dict = {self.gen_input: z}
                    gen_loss_, _ = session.run([self.gen_loss, gen_optimizer], feed_dict = feed_dict)
                    G_losses.append(gen_loss_)
                    
                if epoch % 2 == 0:
                    print('Epoch: %d,  Generator Loss: %.3f,  Discriminator Loss: %.3f' % (epoch, np.mean(G_losses), np.mean(D_losses)))
                self.gen_losses.append(np.mean(G_losses))
                self.dis_losses.append(np.mean(D_losses))
                
            self.save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))    
            print("Model saved in path: %s " % self.save_path)
            self.show_results(session, gen_out)
            
    def show_results(self, session, gen_out):
        z = np.random.uniform(-1, 1, (25, 100))
        feed_dict = {self.gen_input: z}
        test_images = session.run(gen_out, feed_dict = feed_dict)
        print("test images shape: ", len(test_images))
        grid_size = 5
        fig, ax = plt.subplots(grid_size, grid_size, figsize = (5,5))
        for i, j in itertools.product(range(grid_size), range(grid_size)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')
        label = 'Epoch {0}'.format(config.NUM_EPOCHS)
        fig.text(0.5, 0.04, label, ha='center')
        plt.show()