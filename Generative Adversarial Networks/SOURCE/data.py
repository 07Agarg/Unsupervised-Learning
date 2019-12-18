# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:05:32 2019

@author: ashima
"""

import config
from tensorflow.examples.tutorials.mnist import input_data

class DATA():

    def __init__(self):
        self.size = 0
        
    def read(self):
        self.dataset = input_data.read_data_sets(config.DATA_DIR, one_hot = False)
        self.size = len(self.dataset.train.images)
        print(self.size)
        
    def read_test(self):
        test_data_x = self.dataset.test.images
        test_data_y = self.dataset.test.labels
        return test_data_x, test_data_y
        
    def generate_train_batch(self):
        input_, label= self.dataset.train.next_batch(config.BATCH_SIZE)
        return input_, label
