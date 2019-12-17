# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:20 2019

@author: ashima.garg
"""

import os
import tensorflow as tf
import config
import neural_network

class MODEL():
    
    def __init__(self):
        self.gen_input = tf.placeholder(shape = [None, 100], dtype = tf.float32)
        self.dis_input = tf.placeholder(shape = [None, config.IMAGE_SIZE], dtype = tf.float32) 
        
        self.dis_output1 = None            #D(x)
        self.dis_output2 = None            #D(G(z))
        self.gen_loss = None
        self.dis_loss = None
        self.save_path = None
    
    def Generator(self, Z):
        dim = Z.get_shape()[1].value
        hidden1 = neural_network.Generator_Layer(shape = [dim, 256], name = "gn_1", stddev = 0.01, value = 0.1)
        h1 = hidden1.feed_forward_lrelu(Z)
        
        hidden2 = neural_network.Generator_Layer(shape = [256, 512], name = "gn_2", stddev = 0.01, value = 0.1)
        h2 = hidden2.feed_forward_lrelu(h1)
        
        hidden3 = neural_network.Generator_Layer(shape = [512, 1024], name = "gn_3", stddev = 0.01, value = 0.1)
        h3 = hidden3.feed_forward_lrelu(h2)
        
        hidden4 = neural_network.Generator_Layer(shape = [1024, 784], name = "gn_4", stddev = 0.01, value = 0.1)
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
    
    def loss_fun(self, margin = 5.0):
        eps = 1e-2
        self.dis_loss = tf.reduce_mean(-tf.log(self.dis_output1 + eps) - tf.log(1 - self.dis_output2+eps))
        self.gen_loss = tf.reduce_mean(-tf.log(self.dis_output2+eps))
    
    def train(self, data):
        self.loss_fun()
        t_vars = tf.trainable_variables()
        gen_vars = [var for var in t_vars if "generator" in var.name]
        dis_vars = [var for var in t_vars if "discriminator" in var.name]
        
        gen_optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.gen_loss, var_list = gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.dis_loss, var_list = dis_vars)
        
        saver = tf.train.Saver()
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            print("Variables initialized.... ")
            for epoch in range(config.NUM_EPOCHS):
                G_loss = 0
                D_loss = 0
                for batch in range(int(data.size/config.BATCH_SIZE)):
                    batchX, batchY = data.generate_train_batch()
                    z = np.random.normal(0, 1, (config.BATCH_SIZE, 100))
                    #discriminator step
                    feed_dict = {self.dis_input: batchX, self.gen_input = z}
                    dis_loss_, _ = session.run([self.dis_loss, dis_optimizer], feed_dict = feed_dict)
                    
                    #generator step
                    feed_dict = {self.gen_input: z}
                    gen_loss_, _ = session.run([self.gen_loss, gen_optimizer], feed_dict = feed_dict)
                    
                if epoch % 500 == 0:
                    print('Epoch: %d Loss: %.3f' % (epoch, gen_loss_, dis_loss_))
            self.save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))    
            print("Model saved in path: %s " % self.save_path)
            
    def test(self, input_1):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))            
            feed_dict = {self.inputs_1: input_1}
            output = session.run(self.output_1, feed_dict = feed_dict)
            output.tofile(os.path.join(config.OUT_DIR, 'embed.txt'))