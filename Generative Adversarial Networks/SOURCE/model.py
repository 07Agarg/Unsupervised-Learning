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
        self.dis_output1 = None
        self.dis_output2 = None
        self.loss = None
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
        labels = self.labels
        C = tf.constant(margin, name = "C")
        eucd2 = tf.pow(tf.subtract(self.output_1, self.output_2), 2)
        eucd_pos = tf.reduce_sum(eucd2, 1, name = "eucd_pos")
        eucd = tf.sqrt(eucd_pos + 1e-6, name = "eucd")
        eucd_neg = tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2, name = "eucd_neg")
        loss_pos = tf.multiply(labels, eucd_pos, name = "pos_contrastive_loss")
        loss_neg = tf.multiply(tf.subtract(1.0, labels), eucd_neg, name = "neg_conrastive_loss")
        loss = tf.reduce_mean(tf.add(loss_pos, loss_neg), name = "contrastive_loss")
        return loss
    
    def train(self, data):
        self.loss = self.loss_fun()
        optimizer = tf.train.GradientDescentOptimizer(config.LEARNING_RATE).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            print("Variables initialized.... ")
            for epoch in range(config.NUM_EPOCHS):
                batch_X1, batch_X2, batch_label = data.generate_train_batch()
                feed_dict = {self.inputs_1: batch_X1, self.inputs_2: batch_X2, self.labels: batch_label}
                loss_val, _ = session.run([self.loss, optimizer], feed_dict = feed_dict)
                if epoch % 500 == 0:
                    print('Epoch: %d Loss: %.3f' % (epoch, loss_val))
            self.save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))    
            print("Model saved in path: %s " % self.save_path)
            
    def test(self, input_1):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))            
            feed_dict = {self.inputs_1: input_1}
            output = session.run(self.output_1, feed_dict = feed_dict)
            output.tofile(os.path.join(config.OUT_DIR, 'embed.txt'))