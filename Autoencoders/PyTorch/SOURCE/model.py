# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:17:33 2019

@author: Ashima
"""

import os
import utils
import config
import itertools
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable


class Operators():
    def __init__(self, net):
        self.net = net
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = config.LEARNING_RATE)        
        
    def train(self, data):
        losses = []
        for epoch in range(config.NUM_EPOCHS):
            epoch_loss = []
            for batch in range(int(data.size/config.BATCH_SIZE)):
                batchX = data.generate_train_batch()
                batchX_t = Variable(torch.Tensor(batchX).float()).cuda()
                
                #update autoencoder
                self.net.zero_grad()
                out = self.net(batchX_t, True)
                loss_ = self.loss(batchX_t, out)
                epoch_loss.append(loss_.item())
                loss_.backward()
                self.optimizer.step()
                
            print('Epoch: %d,  Loss: %.3f' % (epoch, np.mean(epoch_loss)))
            
            if epoch % 10 == 0:
                self.test(epoch)
            
            losses.append(np.mean(epoch_loss))
        
        #save the model
        self.save_path = os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt")
        torch.save(self.net, self.save_path)
        print("Model saved in path: %s " % self.save_path)
        utils.plot(losses)
        
        #plot results for last epoch
        self.test(epoch)
        
    def test(self, epoch):      
        #self.net.eval()
       if config.DATASET == "MNIST":
           z = Variable(torch.randn(config.TEST_SAMPLES, 16)).cuda()
       elif config.DATASET == "CIFAR":
           z = Variable(torch.randn(config.TEST_SAMPLES, 24, 8, 8)).cuda()
           
       test_images = self.net(z, False).cpu().detach().numpy()
       print("test images shape: ", np.shape(test_images))
       print("test images shape: ", len(test_images))
       grid_size = 5
       fig, ax = plt.subplots(grid_size, grid_size, figsize = (5, 5))
       
       for i, j in itertools.product(range(grid_size), range(grid_size)):
           ax[i, j].get_xaxis().set_visible(False)
           ax[i, j].get_yaxis().set_visible(False)
           
       for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            img = np.transpose(test_images[k], (1, 2, 0))
            ax[i, j].imshow(img)   #.astype('uint8'))
       label = 'Epoch {0}'.format(epoch)
       fig.text(0.5, 0.04, label, ha='center')
       plt.savefig(config.OUT_DIR + 'Generated_Images_GANS_'+str(epoch)+'.jpg')
       plt.show()