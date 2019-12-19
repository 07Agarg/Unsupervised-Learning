# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:19 2019

@author: ashima.garg
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 28*28
BATCH_SIZE = 100

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
USE_PRETRAINED = False
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
TEST_SAMPLES = 25