# -*- coding: utf-8 -*-

#Created on Fri Jun  8 20:43:05 2018

#@author: ashima.garg


import config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def plot_with_labels(embeddings, data):
    plt.figure(figsize=(18, 18))  # in inches
    plot_only = 500
    low_dim_embs = tsne(embeddings[:plot_only, :])
    labels = [data.reverse_dictionary[i] for i in range(plot_only)]
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(os.path.join(config.OUT_DIR, config.RESULT_FILE))


def tsne(embeddings):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(embeddings)
    return low_dim_embs

"""
import os
import config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_with_labels(embeddings, data, plot_only=500):
    plt.figure(figsize=(18, 18))
    low_dim_embeddings = tsne(embeddings[:plot_only, :])
    labels = [data.reverse_dictionary[i] for i in range(plot_only)]
    for i, label in enumerate(labels):
        x, y = low_dim_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    filename = os.path.join(config.OUT_DIR, config.OUT_FILENAME)
    plt.savefig(filename)


def tsne(embeddings):
    tsne = TSNE(perplexity=30,
                n_components=2,
                init='pca',
                n_iter=5000,
                method='exact')
    return tsne.fit_transform(embeddings)
"""