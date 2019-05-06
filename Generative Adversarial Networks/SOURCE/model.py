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
        self.dis_input = tf.placeholder(shape = [None, config.IMAGE_SIZE, config.IMAGE_SIZE, config.DIM], dtype = tf.float32)    
        #self.y = tf.placeholder(shape = [None], dtype = tf.float32)
        self.output_1 = None
        self.output_2 = None
        self.loss = None
        self.save_path = None
        
    """
    def build(self, input_):
        for i in range(len(config.LAYERS)-1):
            hidden = neural_network.Hidden_Layer(config.LAYERS[i], "fc" + str(i))
            input_ = hidden.feed_forward(input_)
        
        outer = neural_network.Outer_Layer(config.LAYERS[-1], "final_fc")
        output_ = outer.feed_forward(input_)
        return output_
    """
    
    def Generator(self, Z):
        hidden1 = neural_network.Generator_Layer()
        h1 = hidden1.feed_forward_lrelu(Z)
        
        hidden2 = neural_network.Generator_Layer()
        h2 = hidden2.feed_forward_lrelu(h1)
        
        hidden3 = neural_network.Generator_Layer()
        h3 = hidden3.feed_forward_tanh(h2)
        
        return h3
    
    def Discriminator(self, labels_, logits_):
        pass
    
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
    
