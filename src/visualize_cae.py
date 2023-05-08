import h5py
import numpy as np
from pyts.image import RecurrencePlot
import tensorflow as tf
import random

from src.clustering_model.ConvolutionalAutoEncoder import ConvolutionalAutoEncoder

import matplotlib.pyplot as plt
from matplotlib import cm

from src.preprocess.load_data import load_periodic_sims

periodic_signal, periodic_labels = load_periodic_sims(0.1)

transformer = RecurrencePlot(threshold=None)
recurrence_plots = transformer.transform(periodic_signal)
recurrence_plots = tf.expand_dims(recurrence_plots, axis=3)
print(recurrence_plots.shape)

test = ConvolutionalAutoEncoder()
optimizer = tf.keras.optimizers.Adam()
test.model.compile(optimizer=optimizer, loss="mse")
test.model.fit(recurrence_plots, recurrence_plots, batch_size=32, epochs=100, verbose=0)
test.model.summary()
images_reconstructed = test.model(recurrence_plots)

selected = random.sample(range(images_reconstructed.shape[0]), 10)

cmap = cm.coolwarm

for count, idx in enumerate(selected):
    data_original = recurrence_plots[idx, :, :, 0]
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

    plt.savefig("../data/images/example_%s.png" % idx, bbox_inches="tight")
