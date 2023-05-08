import numpy as np
from keras import Model
from sklearn.cluster import KMeans

from src.clustering_model.SoftLabels import SoftLabels
from src.clustering_model.ConvolutionalAutoEncoder import ConvolutionalAutoEncoder


class GeneClusteringModel(object):
    def __init__(self, input_shape, num_clusters=10):
        super(GeneClusteringModel, self).__init__()

        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.y_pred = []

        self.cae = ConvolutionalAutoEncoder(input_shape)
        self.soft_labels = SoftLabels(self.num_clusters, name="cluster_labels")

        latent = self.cae.model.get_layer(name="latent").output
        cluster_labels = self.soft_labels(latent)

        self.encoder = Model(inputs=self.cae.model.input, outputs=latent)
        self.model = Model(
            inputs=self.cae.model.input, outputs=[cluster_labels, self.cae.model.output]
        )

    def train_cae(self, x, batch_size=32, epochs=200, optimizer="adam"):
        self.cae.model.compile(optimizer=optimizer, loss="mse")
        self.cae.model.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=0)

    def predict_clusters(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=["kld", "mse"], loss_weights=[1, 1], optimizer="adam"):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(
        self,
        x,
        batch_size=32,
        max_training_steps=2e4,
        tolerance=1e-3,
        update_interval=140,
    ):
        save_interval = x.shape[0] / batch_size * 5

        # Step 1: trait convolutional autoencoder
        self.train_cae(x, batch_size)

        # Step 2: initialize cluster centers using k-means
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20)
        
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name="cluster_labels").set_weights(
            [kmeans.cluster_centers_]
        )

        # Step 3: deep clustering
        index = 0
        for ite in range(int(max_training_steps)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(
                    q
                )  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)

                # check stop criterion
                delta_label = (
                    np.sum(self.y_pred != y_pred_last).astype(np.float32)
                    / self.y_pred.shape[0]
                )
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tolerance:
                    print(f"delta_label: {delta_label} < tolerance: {tolerance}")
                    print(
                        f"Reached tolerance threshold {tolerance}. Stopping training."
                    )
                    
                    kmeans = KMeans(n_clusters=self.num_clusters, n_init=20)
        
                    self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
            
                    #self.inertia = kmeans.inertia_
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(
                    x=x[index * batch_size : :],
                    y=[p[index * batch_size : :], x[index * batch_size : :]],
                )
                index = 0
            else:
                loss = self.model.train_on_batch(
                    x=x[index * batch_size : (index + 1) * batch_size],
                    y=[
                        p[index * batch_size : (index + 1) * batch_size],
                        x[index * batch_size : (index + 1) * batch_size],
                    ],
                )
                index += 1
            ite += 1
