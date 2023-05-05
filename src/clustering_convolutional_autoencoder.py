import random

import h5py
import numpy as np
import pandas as pd
from pyts.image import RecurrencePlot
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.model_selection import StratifiedKFold

from src.clustering_model.GeneClusteringModel import GeneClusteringModel
from src.plotting.latent_pca import plot_cluster_pca
from src.preprocess.load_periodic_sims import load_periodic_sims

periodic_signal, periodic_labels = load_periodic_sims(0.01)

transformer = RecurrencePlot(threshold=None)
recurrence_plots = transformer.transform(periodic_signal)
recurrence_plots = tf.expand_dims(recurrence_plots, axis=3)
print(recurrence_plots.shape)

num_clusters = len(np.unique(periodic_labels))
kf = StratifiedKFold(n_splits=10)
kf.get_n_splits(recurrence_plots)
evaluation_metrics = {
    "ARI": [],
    "AMI": [],
    "Sil": [],
}

for i, (train_index, test_index) in enumerate(
    kf.split(recurrence_plots, periodic_labels)
):
    print(f"Fold {i + 1}")
    x_train = tf.gather(recurrence_plots, indices=train_index)
    x_test = tf.gather(recurrence_plots, indices=test_index)

    y_train = np.squeeze(tf.gather(periodic_labels, indices=train_index).numpy())
    y_test = np.squeeze(tf.gather(periodic_labels, indices=test_index))

    gene_clustering_model = GeneClusteringModel(
        input_shape=recurrence_plots[0].shape, num_clusters=num_clusters
    )

    gene_clustering_model.compile(
        loss=["kld", "mse"], loss_weights=[5, 1], optimizer="adam"
    )
    gene_clustering_model.fit(
        x_train,
        tolerance=1e-3,
        max_training_steps=5000,
        update_interval=100,
    )

    y_pred = gene_clustering_model.predict_clusters(x_test)

    ARI = adjusted_rand_score(y_test, y_pred)
    AMI = adjusted_mutual_info_score(y_test, y_pred)
    sil_score = silhouette_score(gene_clustering_model.encoder.predict(x_test), y_pred)
    evaluation_metrics["ARI"].append(ARI)
    evaluation_metrics["AMI"].append(AMI)
    evaluation_metrics["Sil"].append(sil_score)
    print(
        f"Adjusted Rand Index: {ARI}",
        f"Adjusted Mutual Information: {AMI}",
        f"Silhhouette score: {sil_score}",
    )
    latent_representation = gene_clustering_model.encoder.predict(x_train)
    predicted_clusters = gene_clustering_model.predict_clusters(x_train)

    plt.figure()
    plt.subplot(1, 2, 1)
    plot_cluster_pca(latent_representation, y_train)
    plt.title("True")
    plt.subplot(1, 2, 2)
    plot_cluster_pca(latent_representation, predicted_clusters)
    plt.title("Inferred")
    plt.ylabel("")
    plt.savefig(f"../data/images/latent_pca_{i + 1}.png")

pd.DataFrame.from_dict(evaluation_metrics).to_csv(
    "../data/evaluation/clustering_evaluation.csv"
)
