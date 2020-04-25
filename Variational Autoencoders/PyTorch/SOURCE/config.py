# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:39:16 2020

@author: Ashima
"""

import os
import torch

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_DIR = "train"
TEST_DIR = "test"

DATASET = "MNIST"
NETWORK_TYPE = "LINEAR"

# DATA INFORMATION
IMG_DIM = 28
CHANNELS = 1
IMAGE_SIZE = IMG_DIM*IMG_DIM*CHANNELS
BATCH_SIZE = 16
LATENT_DIM = 32
NUM_CLASSES = 10

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
                                                                        'cpu')

# TRAINING INFORMATION
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
TEST_SAMPLES = 25
DROPOUT = 0.1
