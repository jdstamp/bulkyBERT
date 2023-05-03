import h5py
import numpy as np
from pyts.image import RecurrencePlot
import tensorflow as tf
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold

from src.clustering_model.ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
from src.clustering_model.GeneClusteringModel import GeneClusteringModel

data_sim = h5py.File("../data/data_simulated/sim_erdosrenyi.h5", "r")
train_examples = data_sim.get("expression/data")[:]
train_labels = data_sim.get("expression/labels")[:]
train_examples = np.transpose(train_examples, (0, 2, 1))

transformer = RecurrencePlot(threshold=None)
train_images = []
for sample in train_examples:
    padded = np.pad(
        sample, ((0, 0), (0, 0))
    )  # in case the image is not a power of 2, fix here
    train_images.append(transformer.fit_transform(padded[:1, :]))
train_images = np.array(train_images)
train_images = np.reshape(train_images, (-1, 16, 16, 1))
train_labels = np.reshape(train_labels, (-1, 1))
print(train_images.shape)

test = ConvolutionalAutoEncoder()
optimizer = tf.keras.optimizers.Adam()
test.model.compile(optimizer=optimizer, loss="mse")
test.model.fit(train_images, train_images, batch_size=32, epochs=10, verbose=0)
test.model.summary()


num_clusters = len(np.unique(train_labels))
accuracy = tf.keras.metrics.Accuracy()
kf = StratifiedKFold(n_splits=10)
kf.get_n_splits(train_images)
my_counter = 1
for train_index, test_index in kf.split(train_images, train_labels):
    x_train = train_images[train_index]
    x_test = train_images[test_index]

    y_train = train_labels[train_index]
    y_test = train_labels[test_index]

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

    acc = accuracy(y_test, y_pred)
    sil_score = silhouette_score(gene_clustering_model.encoder.predict(x_test), y_pred)
    print(f"Accuracy: {acc}, Silhhouette score: {sil_score}")
