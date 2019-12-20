# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:55:58 2019

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

class Generator_Model(nn.Module):
    def __init__(self):
        super(Generator_Model, self).__init__()
        self.hidden1 = nn.Linear(100, 256)
        self.hidden2 = nn.Linear(256, 512)
        self.hidden3 = nn.Linear(512, 1024)
        self.hidden4 = nn.Linear(1024, 784)
        
    def forward(self, X):
        out = nn.functional.leaky_relu(self.hidden1(X), negative_slope = 0.2, inplace = False)
        out = nn.functional.leaky_relu(self.hidden2(out), negative_slope = 0.2, inplace = False)
        out = nn.functional.leaky_relu(self.hidden3(out), negative_slope = 0.2, inplace = False)
        out = nn.functional.tanh(self.hidden4(out))
        return out.cuda()

class Discriminator_Model(nn.Module):
    def __init__(self):
        super(Discriminator_Model, self).__init__()
        self.hidden1 = nn.Linear(784, 1024)
        self.hidden2 = nn.Linear(1024, 512)
        self.hidden3 = nn.Linear(512, 256)
        self.hidden4 = nn.Linear(256, 1)
        
    def forward(self, X):
        out = nn.functional.dropout(nn.functional.leaky_relu(self.hidden1(X), negative_slope = 0.2, inplace = False), p = 0.3)
        out = nn.functional.dropout(nn.functional.leaky_relu(self.hidden2(out), negative_slope = 0.2, inplace = False), p = 0.3)
        out = nn.functional.dropout(nn.functional.leaky_relu(self.hidden3(out), negative_slope = 0.2, inplace = False), p = 0.3)
        out = nn.functional.sigmoid(self.hidden4(out))
        return out.cuda()

class Operators():
    def __init__(self, gen_net, dis_net):
        self.gen_net = gen_net
        self.dis_net = dis_net
        self.loss = nn.BCELoss()
        self.gen_optimizer = torch.optim.Adam(self.gen_net.parameters(), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
        self.dis_optimizer = torch.optim.Adam(self.dis_net.parameters(), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
        
    def train(self, data):
        gen_losses = []
        dis_losses = []
        for epoch in range(config.NUM_EPOCHS):
            G_losses = []
            D_losses = []
            for batch in range(int(data.size/config.BATCH_SIZE)):
                batchX, batchY = data.generate_train_batch()
                batchX_t = Variable(torch.Tensor(batchX).float()).cuda()
                z = Variable(torch.randn(config.BATCH_SIZE, 100)).cuda()
                real_labels = Variable(torch.ones(config.BATCH_SIZE)).cuda()
                fake_labels = Variable(torch.zeros(config.BATCH_SIZE)).cuda()
                
                #update discriminator
                self.dis_net.zero_grad()       #check net.zero_grad() or optimizer.zero_grad()
                out1 = self.dis_net(batchX_t)  #D(x)
                gen_out = self.gen_net(z)
                out2 = self.dis_net(gen_out)   #D(G(z))  
                
                dis_loss_1 = self.loss(out1, real_labels)
                dis_loss_2 = self.loss(out2, fake_labels)
                dis_loss_ = dis_loss_1 + dis_loss_2
                
                D_losses.append(dis_loss_.item())
                dis_loss_.backward()
                self.dis_optimizer.step()
                
                #update generator
                self.gen_net.zero_grad()
                gen_out = self.gen_net(z)
                out2 = self.dis_net(gen_out)   #D(G(z))
                gen_loss_ = self.loss(out2, real_labels)
                G_losses.append(gen_loss_.item())
                gen_loss_.backward()
                self.gen_optimizer.step()
                
            if epoch % 2 == 0:
                print('Epoch: %d,  Generator Loss: %.3f,  Discriminator Loss: %.3f' % (epoch, np.mean(G_losses), np.mean(D_losses)))
            
            if epoch % 10 == 0:
                self.test(epoch)
            
            gen_losses.append(np.mean(G_losses))
            dis_losses.append(np.mean(D_losses))
        
        self.save_path = os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt")
        torch.save(self.gen_net, self.save_path)
        print("Model saved in path: %s " % self.save_path)
        utils.plot(gen_losses, dis_losses)
        self.test(epoch)
        
    def test(self, epoch):      
       z = Variable(torch.randn(config.TEST_SAMPLES, 100)).cuda()
       test_images = self.gen_net(z).cpu().detach().numpy()
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
            ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')
       label = 'Epoch {0}'.format(epoch)
       fig.text(0.5, 0.04, label, ha='center')
       plt.savefig(config.OUT_DIR + 'Generated_Images_GANS_'+str(epoch)+'.jpg')
       plt.show()