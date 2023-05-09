import os

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

from src.clustering_model.GeneClusteringModel import GeneClusteringModel
from src.plotting.latent_pca import plot_cluster_pca
from src.preprocess.load_data import load_varoquaux_data

def main():
    gene_expression, all_labels = load_varoquaux_data()
    labels = np.array(
        [[string.decode("utf-8") for string in row] for row in all_labels.numpy()]
    )
    labels = labels[:, 1] #np.apply_along_axis(lambda x: "-".join(x), 1, labels[:, [1,3]])

    transformer = RecurrencePlot(threshold=None)
    recurrence_plots = transformer.transform(gene_expression)
    recurrence_plots = tf.expand_dims(recurrence_plots, axis=3)

    evaluation_metrics = {
        "ARI": [],
        "AMI": [],
        "Sil": [],
    }
    print(recurrence_plots.shape)
    for num_clusters in range(3, 9, 1):
        model_file = f"../data/models/varoquaux_model_{num_clusters}_clusters"
        encoder_file = f"../data/models/varoquaux_encoder_{num_clusters}_clusters"
        print(f"Number of clusters: {num_clusters}")
        if not os.path.exists(model_file):
            train_model(recurrence_plots, num_clusters, model_file, encoder_file)
        eval_model(recurrence_plots, labels, num_clusters, model_file, encoder_file, evaluation_metrics)
    pd.DataFrame.from_dict(evaluation_metrics).to_csv(
        f"../data/evaluation/varoquaux.csv"
    )


def train_model(recurrence_plots, num_clusters, model_file, encoder_file):
    gene_clustering_model = GeneClusteringModel(
        input_shape=recurrence_plots[0].shape, num_clusters=num_clusters
    )

    gene_clustering_model.compile(
        loss=["kld", "mse"], loss_weights=[1, 1], optimizer="adam"
    )
    gene_clustering_model.fit(
        recurrence_plots,
        tolerance=1e-3,
        max_training_steps=1000,
        update_interval=100,
    )

    gene_clustering_model.model.save(model_file)
    gene_clustering_model.encoder.save(encoder_file)


def eval_model(recurrence_plots, labels, num_clusters, model_file, encoder_file, evaluation_metrics):
    model = tf.keras.models.load_model(model_file)
    encoder = tf.keras.models.load_model(encoder_file)
    latent_representation = encoder.predict(recurrence_plots)
    predicted_clusters = predict_clusters(model, recurrence_plots)
    label_codes = pd.Categorical(labels).codes

    ARI = adjusted_rand_score(label_codes, predicted_clusters)
    AMI = adjusted_mutual_info_score(label_codes, predicted_clusters)
    sil_score = silhouette_score(encoder.predict(recurrence_plots), predicted_clusters)
    evaluation_metrics["ARI"].append(ARI)
    evaluation_metrics["AMI"].append(AMI)
    evaluation_metrics["Sil"].append(sil_score)
    print(
        f"Adjusted Rand Index: {ARI}",
        f"Adjusted Mutual Information: {AMI}",
        f"Silhhouette score: {sil_score}",
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_cluster_pca(latent_representation, labels, ax)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Sorghum")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid("on")
    fig.savefig(
        f"../data/images/varoquaux_{num_clusters}_clusters.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_cluster_pca(latent_representation, predicted_clusters, ax)
    ax.set_title("Inferred Clusters")
    ax.grid("on")
    fig.savefig(
        f"../data/images/varoquaux_{num_clusters}_clusters_inferred.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )


def predict_clusters(model, x):
    q, _ = model.predict(x, verbose=0)
    return q.argmax(1)


if __name__ == "__main__":
    main()
