import pandas as pd
import tensorflow as tf

train_examples = pd.read_csv("/Users/jds/git/bulkyBERT/train_data.csv.gz",
                             compression="gzip",
                             index_col=0)
                             
train_labels = train_examples["label"]
train_examples = train_examples.drop(["label"], axis = 1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
print(train_dataset)
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 3

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(5)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
model.fit(train_dataset, epochs=10)
