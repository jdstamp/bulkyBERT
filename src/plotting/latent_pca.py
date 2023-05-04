import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats


def plot_cluster_pca(X, y):
    X = stats.zscore(X)
    pca = PCA()
    x_new = pca.fit_transform(X)
    fig = plt.figure()
    plt.scatter(x_new[:, 0], x_new[:, 1], c=y)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    return fig
