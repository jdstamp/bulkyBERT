import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE


def plot_cluster_tsne(X, y):
    X = stats.zscore(X)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    fig = plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y)
    plt.xlabel("tsne-1")
    plt.ylabel("tsne-2")
    return fig
