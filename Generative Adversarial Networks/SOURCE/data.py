# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:05:32 2019

@author: ashima
"""

from tensorflow.examples.tutorials.mnist import input_data
import config


class DATA():

    def __init__(self):
        self.data_index = 0
        self.dataset = None

    def read(self):
        self.dataset = input_data.read_data_sets(config.DATA_DIR, one_hot = False)
        
    def read_test(self):
        test_data_x = self.dataset.test.images
        test_data_y = self.dataset.test.labels
        return test_data_x, test_data_y
        
    def generate_train_batch(self):
        input_1, label_1 = self.dataset.train.next_batch(config.BATCH_SIZE)
        input_2, label_2 = self.dataset.train.next_batch(config.BATCH_SIZE)
        label = (label_1 == label_2).astype('float32')
        return input_1, input_2, label
