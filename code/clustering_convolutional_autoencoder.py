import h5py
import numpy as np
from pyts.image import RecurrencePlot
import tensorflow as tf

from code.clustering_model.ConvolutionalAutoEncoder import ConvolutionalAutoEncoder

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
print(train_images.shape)

test = ConvolutionalAutoEncoder()
optimizer = tf.keras.optimizers.Adam()
test.cae.compile(optimizer=optimizer, loss="mse")
test.cae.fit(train_images, train_images, batch_size=32, epochs=10, verbose=0)
test.cae.summary()
