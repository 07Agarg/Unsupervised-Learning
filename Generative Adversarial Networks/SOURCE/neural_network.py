# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:20 2019

@author: ashima.garg
"""

import tensorflow as tf

class Layer():

    def __init__(self, shape, name, stddev, value):
        self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        self.biases = tf.Variable(tf.constant(value=value, shape=[shape[-1]]))

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
        super(Discriminator_Layer).__init__(shape, name, stddev, value)
    
    def feed_forward_1(self, input_data):
        output = tf.nn.dropout(tf.nn.leaky_relu(tf.add(tf.matmul(input_data, self.weights), self.biases)), )
        return output
        
    def feed_forward_2(self, input_data):
        output = tf.nn.sigmoid(tf.add(tf.matmul(input_data, self.weights), self.biased))
        return output
    
    
"""    
        initer = tf.truncated_normal_initializer(stddev = 0.01, mean = 0)
        self.weight = tf.get_variable(name = name+"W", shape=shape, initializer = initer, dtype = tf.float32)
        self.bias = tf.get_variable(name = name+"b", shape = [shape[1]], initializer = tf.constant_initializer(0.01))
        
    def feed_forward(self, input_):
        output_ = tf.nn.relu(tf.add(tf.matmul(input_, self.weight), self.bias))
        return output_
        
class Outer_Layer:
    
    def __init__(self, shape, name):
        initer = tf.truncated_normal_initializer(stddev = 0.01, mean = 0)
        self.weight = tf.get_variable(name = name+"W", shape=shape, initializer = initer, dtype = tf.float32)
        self.bias = tf.get_variable(name = name+"b", shape = [shape[1]], initializer = tf.constant_initializer(0.01))
        
    def feed_forward(self, input_):
        output_ = tf.add(tf.matmul(input_, self.weight), self.bias)
        return output_
"""