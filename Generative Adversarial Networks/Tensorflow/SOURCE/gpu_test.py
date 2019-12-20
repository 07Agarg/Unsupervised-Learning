# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 05:56:09 2019

@author: Ashima
"""

import tensorflow as tf


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
    
tf = tf.Session(config=tf.ConfigProto(log_device_placement=True))
tf.list_devices()