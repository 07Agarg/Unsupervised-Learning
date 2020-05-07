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
import pandas as pd
import seaborn as sns
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize


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


def plot_latent_tsne(data, net, plot_type="TRAIN"):
    """Plot tsne of latent space."""
    net.load_state_dict(torch.load(model_path()))
    net.eval()
    with torch.no_grad():
        if config.DATASET == "MNIST":
            x = data.dataset.data.type(torch.FloatTensor)[:10000]
            if config.NETWORK_TYPE == "CNN":
                x = x.unsqueeze(1)
            # x = x.view(x.size(0), -1)
            # x = x.view(x.size(0), 1, x.size(2), x.size(3))
        y = data.dataset.targets[:10000]
    encoded, _ = net(x.to(config.device), encode=True, decode=False)
    X = encoded[0].view(encoded[0].size()[0], -1).cpu().detach().numpy()
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    print('Size of the dataframe: {}'.format(df.shape))
    data_subset = df[feat_cols].values
    plot_tsne(data_subset, df, "LatentTSNE_"+plot_type)


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


def plot_roc(y_true, y_pred, string):
    print("y_test shape: ", y_true.shape)
    print("y score shape: ", y_pred.shape)
    y_true = label_binarize(y_true, classes=config.CLASS_LABELS)
    print("ROC Curve")
    fpr = dict()
    tpr = dict()
    for i in range(config.NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    plt.figure()
    colors = sns.color_palette("hsv", config.NUM_CLASSES)
    for i in range(config.NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], label='ROC Curve Class - ' + str(i),
                 color=colors[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot for' + str(string))
    plt.legend(loc='lower right')
    plt.savefig(config.OUT_DIR + 'ROC_PLOT '+string+".jpg")
    plt.show()


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


def visualize_images(test_images, epoch, label):
    """Visualize reconstructed images and generated images
    from randomly sampled values from the latent space."""
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
                        (config.IMG_DIM, config.IMG_DIM)),
                        cmap='gray')
    # label = label + 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, label + str(epoch), ha='center')
    plt.savefig(config.OUT_DIR + label + str(epoch) + '.jpg')


def generate_images_helper(net, epoch, label="After_"):
    """Helper function to visualize generated images from
    randomly sampled values from the latent space."""
    net.eval()
    with torch.no_grad():
        # if config.NETWORK_TYPE == "LINEAR":
        z = torch.randn(config.TEST_SAMPLES, config.LATENT_DIM).to(
            config.device)
        _, test_images = net(z, encode=False, decode=True)
        test_images = test_images.cpu().detach().numpy()
        net.train()
        if label != "Before_training_":
            label = "Generated_Images_"
        else:
            label = label + "Generated_Images_"
        visualize_images(test_images, epoch, label)


def reconstruct_images_helper(data, net, epoch, label="After_"):
    """Helper function to visualize reconstructions."""
    net.eval()
    with torch.no_grad():
        X, _ = next(iter(data.reconstruct_dataloader))
        _, test_images = net(torch.Tensor(X.float()).to(config.device),
                             encode=True, decode=True)
        test_images = test_images.cpu().detach().numpy()
        net.train()

        if label != "Before_training_":
            plot_label = "Original_Images_"
        else:
            plot_label = label + "Original_Images_"
        visualize_images(X, epoch, plot_label)

        if label != "Before_training_":
            plot_label = "Reconstructed_Images_"
        else:
            plot_label = label + "Reconstructed_Images_"
        visualize_images(test_images, epoch, plot_label)


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()


def convert_to_list(data):
    """Convert list of lists to list."""
    final_list = []
    for i in data:
        try:
            for j in i:
                final_list.append(j)
        except:
            print("ERROR")
    final_list = np.array(final_list)
    return final_list
