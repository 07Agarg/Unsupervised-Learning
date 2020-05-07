# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:48:23 2020

@author: Ashima
"""

import utils
import time
import config
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


class vaeLoss(torch.nn.Module):
    def __init__(self):
        super(vaeLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x_orig, x_recon, mu, logvar):
        if config.NETWORK_TYPE == "CNN":
            x_recon = x_recon.view(-1, 784)
        bce_loss = F.binary_cross_entropy(x_recon, x_orig.view(-1, 784),
                                          reduction='sum')
        kldiv_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce_loss, kldiv_loss


class Operators():

    def __init__(self, net):
        self.net = net
        self.loss = vaeLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                    lr=config.LEARNING_RATE,
                                    betas=(0.9, 0.999))

        self.writer = SummaryWriter()

    def train(self, train_data, test_data):
        losses = []
        start_time = time.time()
        self.net.train()
        utils.generate_images_helper(self.net, epoch=0,
                                     label="Before_training_")
        utils.reconstruct_images_helper(test_data, self.net,
                                        epoch=0, label="Before_training_")
        for epoch in range(config.NUM_EPOCHS):
            epoch_loss = []
            kldiv_epoch_loss = []
            bce_epoch_loss = []
            self.net.train()
            for batch in range(int(len(train_data.dataloader.dataset)/config.BATCH_SIZE)):
                X_batch, _ = train_data.load_batch()

                # Update autoencoder
                self.net.zero_grad()
                (z_mu, z_var), x_recon = self.net(X_batch, encode=True,
                                                  decode=True)
                bce_loss, kldiv_loss = self.loss(X_batch, x_recon, z_mu, z_var)
                batch_loss = bce_loss + kldiv_loss
                epoch_loss.append(batch_loss.item())
                kldiv_epoch_loss.append(kldiv_loss.item())
                bce_epoch_loss.append(bce_loss.item())
                batch_loss.backward()
                self.optimizer.step()

            print('Epoch: %d,  Loss: %.3f' % (epoch, np.mean(epoch_loss)))
            self.writer.add_scalar('Total_Loss', np.mean(epoch_loss), epoch)
            self.writer.add_scalar('Kl_divergence_Loss',
                                   np.mean(kldiv_epoch_loss), epoch)
            self.writer.add_scalar('bce_loss', np.mean(bce_epoch_loss), epoch)

            if epoch % 30 == 0 or epoch == 1:
                print("In epoch 1")
                utils.generate_images_helper(self.net, epoch)
                utils.reconstruct_images_helper(test_data, self.net, epoch)

            losses.append(np.mean(epoch_loss))

        self.writer.close()
        # Save the model
        torch.save(self.net.state_dict(), utils.model_path())
        utils.plot_loss(losses)

        print("Total time taken: ", time.time() - start_time)

        # Plot results for last epoch
        utils.generate_images_helper(self.net, epoch+1)
        utils.reconstruct_images_helper(test_data, self.net, epoch+1)
