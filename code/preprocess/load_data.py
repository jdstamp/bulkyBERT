import tensorflow as tf
import h5py
import numpy as np

hf = h5py.File("/Users/jds/git/bulkyBERT/myhdf5file.h5", "r")
train_examples = hf.get("expression/data")[:]
train_labels = hf.get("expression/labels")[:]

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
print(train_dataset)
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 3


train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10, 100)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
model.fit(train_dataset, epochs=10)
