import random
import h5py
import numpy as np
import pandas as pd

from pyts.image import RecurrencePlot
from matplotlib import pyplot as plt
from matplotlib import cm

import tensorflow as tf
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats

# might need to adjust file paths to run
from src.clustering_model.GeneClusteringModel import GeneClusteringModel
from src.plotting.latent_pca import plot_cluster_pca
from src.plotting.latent_pca import plot_cluster_pca_real_data


data = h5py.File("data/shoemaker.h5", "r")

mouse_data = data.get("expression/data")
mouse_data = np.transpose(mouse_data, (1, 0))

gene_labels = data.get("expression/gene_labels")[:]
group_labels = data.get("expression/group_labels")[:]

transformer = RecurrencePlot(threshold=None)
recurrence_plots = transformer.transform(mouse_data)
recurrence_plots = tf.expand_dims(recurrence_plots, axis=3)

kf = StratifiedKFold(n_splits=10)
kf.get_n_splits(recurrence_plots)
evaluation_metrics = {
    "ARI": [],
    "AMI": [],
    "Sil": [],
}

train_images = recurrence_plots
train_labels = [lab.decode('UTF-8') for lab in group_labels]

LE = LabelEncoder()

train_labels_int = LE.fit_transform(train_labels)

sil_per_k = {}
for num_clusters in [2,3,4,5,6,7,8]:
    gene_clustering_model = GeneClusteringModel(
        input_shape=recurrence_plots[0].shape, num_clusters=num_clusters
    )

    gene_clustering_model.compile(
        loss=["kld", "mse"], loss_weights=[5, 1], optimizer="adam"
    )
    gene_clustering_model.fit(
        train_images,
        tolerance=1e-3,
        max_training_steps=5000,
        update_interval=100,
    )

    y_pred = gene_clustering_model.predict_clusters(train_images)

    ARI = adjusted_rand_score(train_labels_int, y_pred)
    AMI = adjusted_mutual_info_score(train_labels_int, y_pred)
    sil_score = silhouette_score(gene_clustering_model.encoder.predict(train_images), y_pred)
    evaluation_metrics["ARI"].append(ARI)
    evaluation_metrics["AMI"].append(AMI)
    evaluation_metrics["Sil"].append(sil_score)
    print(
        f"Adjusted Rand Index: {ARI}",
        f"Adjusted Mutual Information: {AMI}",
        f"Silhhouette score: {sil_score}",
    )

    sil_per_k[num_clusters] = sil_score

    latent_representation = gene_clustering_model.encoder.predict(train_images)
    predicted_clusters = gene_clustering_model.predict_clusters(train_images)

    X = stats.zscore(latent_representation)
    pca = PCA()
    x_new = pca.fit_transform(X)

    cdict = {'C':'blue', 'K':'orange', 'M':'khaki', 'VH':'red', 'VL':'yellow'}

    fig, ax = plt.subplots()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    for g in np.unique(train_labels):
        ix = [element == g for element in train_labels]
        ax.scatter(x_new[ix, 0], x_new[ix, 1], c = cdict[g], label = g)

    ax.legend()
    plt.title("2D Representation of latent clusters")

    plt.savefig(f"data/images/mouse_pca_"+str(num_clusters)+".png")

images_reconstructed = gene_clustering_model.cae.model(train_images)
selected = random.sample(range(train_images.shape[0]), 10)

cmap = cm.coolwarm
for count, idx in enumerate(selected):
    data_original = train_images[idx, :, :, 0]
    data_reconstruct = images_reconstructed[idx, :, :, 0]

    all_data = np.append(data_original, data_reconstruct)
    all_data = np.reshape(all_data, -1)

    zmax = np.max(all_data)
    zmin = np.min(all_data)

    plt.subplot(1, 2, 1)
    plt.imshow(data_original, cmap=cmap, vmin=zmin, vmax=zmax)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(data_reconstruct, cmap=cmap, vmin=zmin, vmax=zmax)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.savefig("data/images/mouse_reconstruction%s.png" % idx, bbox_inches="tight")

lists = sorted(sil_per_k.items())

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.xlabel("Number of Clusters")
plt.ylabel("Silhhouette score")

plt.plot(x, y)
plt.show()
