import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats


def plot_cluster_pca(X, y, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
    X = stats.zscore(X)
    pca = PCA()
    x_new = pca.fit_transform(X)
    ax.scatter(x_new[:, 0], x_new[:, 1], c=y, **plt_kwargs)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return ax
