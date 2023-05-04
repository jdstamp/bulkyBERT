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

from src.clustering_model.ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
from src.clustering_model.GeneClusteringModel import GeneClusteringModel
from src.plotting.latent_pca import plot_cluster_pca
from src.plotting.latent_tsne import plot_cluster_tsne

data_sim = h5py.File("../data/data_simulated/sim_gene_expression.h5", "r")
train_examples = data_sim.get("expression/data")
train_labels = data_sim.get("expression/labels")
train_examples = np.transpose(train_examples[:], (0, 2, 1))
train_labels = np.transpose(train_labels[:], (1, 0))
train_examples = tf.reshape(train_examples, (-1, 16))
train_labels = tf.reshape(train_labels, (-1, 1))

idx = random.sample(range(len(train_labels)), 100)
train_examples = tf.gather(train_examples, indices=idx)
train_labels = tf.gather(train_labels, indices=idx)

transformer = RecurrencePlot(threshold=None)
train_images = transformer.transform(train_examples)
train_images = tf.expand_dims(train_images, axis=3)
print(train_images.shape)

test = ConvolutionalAutoEncoder()
optimizer = tf.keras.optimizers.Adam()
test.model.compile(optimizer=optimizer, loss="mse")
test.model.fit(train_images, train_images, batch_size=32, epochs=10, verbose=0)
test.model.summary()

num_clusters = len(np.unique(train_labels))
kf = StratifiedKFold(n_splits=10)
kf.get_n_splits(train_images)
evaluation_metrics = {
    "ARI": [],
    "AMI": [],
    "Sil": [],
}
for i, (train_index, test_index) in enumerate(kf.split(train_images, train_labels)):
    print(f"Fold {i}")
    x_train = tf.gather(train_images, indices=train_index)
    x_test = tf.gather(train_images, indices=test_index)

    y_train = np.squeeze(tf.gather(train_labels, indices=train_index).numpy())
    y_test = np.squeeze(tf.gather(train_labels, indices=test_index))

    gene_clustering_model = GeneClusteringModel(
        input_shape=train_images[0].shape, num_clusters=num_clusters
    )

    gene_clustering_model.compile(
        loss=["kld", "mse"], loss_weights=[0.1, 1], optimizer="adam"
    )
    gene_clustering_model.fit(
        x_train,
        tolerance=1e-3,
        max_training_steps=1000,
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
    fig = plot_cluster_pca(gene_clustering_model.encoder.predict(x_train), y_train)
    plt.savefig(f"../data/images/latent_pca_{i}.png", bbox_inches="tight")
    fig = plot_cluster_tsne(gene_clustering_model.encoder.predict(x_train), y_train)
    plt.savefig(f"../data/images/latent_tsne_{i}.png", bbox_inches="tight")

pd.DataFrame.from_dict(evaluation_metrics).to_csv(
    "../data/evaluation/clustering_evaluation.csv"
)
