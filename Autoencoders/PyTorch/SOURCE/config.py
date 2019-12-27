# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:17:33 2019

@author: Ashima
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_DIR = "train"
TEST_DIR = "test"

DATASET = "CIFAR"          # Change this to MNIST for MNIST Dataset, and CIFAR to CIFAR10

# DATA INFORMATION
IMAGE_SIZE = 28*28
IMG_DIM = 28
BATCH_SIZE = 128

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
TEST_SAMPLES = 25