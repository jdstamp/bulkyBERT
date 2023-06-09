import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats


def plot_cluster_pca(X, y, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
    X = stats.zscore(X)
    pca = PCA()
    x_new = pca.fit_transform(X)
    for g in np.unique(y):
        ix = [element == g for element in y]
        ax.scatter(x_new[ix, 0], x_new[ix, 1], label=g)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return ax
