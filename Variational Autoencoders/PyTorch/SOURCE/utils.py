# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:39:16 2020

@author: Ashima
"""

import os
import time
import config
import torch
import itertools
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_loss(loss_list):
    """Function to plot loss curve."""
    plt.figure()
    markers = ['.', 'o']
    colors = ['r', 'b']
    x = np.arange(config.NUM_EPOCHS)
    plt.plot(np.asarray(x), np.asarray(loss_list), label="loss",
             color=colors[0], marker=markers[0])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(config.OUT_DIR + 'LossCurve.jpg')
    plt.show()


def plot_latent_tsne(data, net):
    """Plot tsne of latent space."""
    net.load_state_dict(torch.load(model_path()))
    net.eval()
    with torch.no_grad():
        if config.DATASET == "MNIST":
            x = data.dataset.data.type(torch.FloatTensor)
            x = x.view(x.size(0), -1)
        y = data.dataset.targets
    encoded, _ = net(x.to(config.device), encode=True, decode=False)
    X = encoded[0].view(encoded[0].size()[0], -1).cpu().detach().numpy()
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    print('Size of the dataframe: {}'.format(df.shape))
    data_subset = df[feat_cols].values
    plot_tsne(data_subset, df, "LatentTSNE")


def plot_tsne(data_subset, df, string):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hsv", config.NUM_CLASSES),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig(config.OUT_DIR + string + '_' + config.DATASET + '.jpg')


def model_path():
    """Return autoencoder path."""
    path = os.path.join(config.MODEL_DIR,
                        "model" +
                        str(config.BATCH_SIZE) +
                        "_" +
                        str(config.NUM_EPOCHS) +
                        "_" +
                        str(config.DATASET) +
                        ".pt")
    print("model path: ", path)
    return path


def visualize_images(net, epoch):
    """
    Visualize the images generated using the trained decoder from the
    normally distributed random samples.
    """
    # if config.NETWORK_TYPE == "CNN":
    #     z = Variable(torch.randn(config.TEST_SAMPLES, 48, 4, 4)).to(
    #         config.device)
    net.eval()
    with torch.no_grad():
        if config.NETWORK_TYPE == "LINEAR":
            z = torch.randn(config.TEST_SAMPLES, config.LATENT_DIM).to(
                config.device)
        _, test_images = net(z, encode=False, decode=True)
        test_images = test_images.cpu().detach().numpy()
        net.train()
        grid_size = 5
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in itertools.product(range(grid_size), range(grid_size)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        for k in range(config.TEST_SAMPLES):
            i = k // grid_size
            j = k % grid_size
            ax[i, j].cla()
            ax[i, j].imshow(np.reshape(test_images[k],
                            (config.IMG_DIM, config.IMG_DIM)), cmap='gray')
        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(config.OUT_DIR + 'Generated_Images_'+str(epoch)+'.jpg')
        # plt.show()