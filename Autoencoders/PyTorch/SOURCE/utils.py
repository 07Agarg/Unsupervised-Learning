# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:17:33 2019

@author: Ashima
"""

import config
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def plot(loss_list):
    markers = ['.', 'o']
    colors = ['r', 'b']
    x = np.arange(config.NUM_EPOCHS)
    plt.plot(np.asarray(x), np.asarray(loss_list), label = "loss", color = colors[0], marker = markers[0])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(config.OUT_DIR + 'LossCurve.jpg')
    plt.show()