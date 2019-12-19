# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:15:19 2019

@author: ashima.garg
"""

import config
import numpy as np
import matplotlib.pyplot as plt

def plot(gen_loss_list, dis_loss_list):
    #markers = ['.', 'o', 'v', '^', '<']
    #colors = ['r', 'b', 'g', 'c', 'm']
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