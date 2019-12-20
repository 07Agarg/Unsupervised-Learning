# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:55:58 2019

@author: Ashima
"""

#References: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

import config
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.)

def plot(gen_loss_list, dis_loss_list):
    markers = ['.', 'o']
    colors = ['r', 'b']
    x = np.arange(config.NUM_EPOCHS)
    plt.plot(np.asarray(x), np.asarray(gen_loss_list), label = "Gen_loss", color = colors[0], marker = markers[0])
    plt.plot(np.asarray(x), np.asarray(dis_loss_list), label = "Dis_loss", color = colors[1], marker = markers[1])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(config.OUT_DIR + 'LossCurve.jpg')
    plt.show()