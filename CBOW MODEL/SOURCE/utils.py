# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:43:05 2018

@author: ashima.garg

"""

import config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import data
import os

def plot_with_labels(embeddings, labels):
    low_dim_embs = tsne(embeddings)
    plt.figure(figsize=(18, 18))  # in inches
    labels = [data.reverse_dictionary[i] for i in range(plot_only = 500)]
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(os.path.join(config.OUT_DIR, config.RESULT_FILE))


def tsne(embeddings):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    return low_dim_embs
