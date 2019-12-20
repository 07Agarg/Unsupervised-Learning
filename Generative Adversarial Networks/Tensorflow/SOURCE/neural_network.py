# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:20 2019

@author: ashima.garg
"""

import tensorflow as tf

class Layer():

    def __init__(self, shape, name, stddev, value):
#        initer = tf.truncated_normal_initializer(stddev = stddev, mean = 0)
        initer = tf.random_normal_initializer(stddev = stddev, mean = 0)
        self.weights = tf.get_variable(name = name+"W", shape=shape, initializer = initer, dtype = tf.float32)
        self.biases = tf.get_variable(name = name+"b", shape = [shape[1]], initializer = tf.constant_initializer(value))
        
    def feed_forward(self, input_data):
        raise NotImplementedError

class Generator_Layer(Layer):
    
    def __init__(self, shape, name, stddev, value):
        super(Generator_Layer, self).__init__(shape, name, stddev, value)
    
    def feed_forward_lrelu(self, input_data):
        output = tf.nn.leaky_relu(tf.add(tf.matmul(input_data, self.weights), self.biases))
        return output
    
    def feed_forward_tanh(self, input_data):
        output = tf.nn.tanh(tf.add(tf.matmul(input_data, self.weights), self.biases))
        return output
    
class Discriminator_Layer(Layer):
    
    def __init__(self, shape, name, stddev, value):
        super(Discriminator_Layer, self).__init__(shape, name, stddev, value)
    
    def feed_forward_1(self, input_data, dropout):
        output = tf.nn.dropout(tf.nn.leaky_relu(tf.add(tf.matmul(input_data, self.weights), self.biases)), keep_prob=dropout)
        return output
        
    def feed_forward_2(self, input_data, dropout):
        output = tf.nn.sigmoid(tf.add(tf.matmul(input_data, self.weights), self.biases))
        return output