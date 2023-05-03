import tensorflow as tf


class ConvolutionalAutoEncoder(tf.keras.Model):
    def __init__(self, target_shape=(16, 16, 1), latent_dim=10):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim, name="latent"),
                tf.keras.layers.Dense(
                    128
                    * self.caclulate_shape(target_shape)
                    * self.caclulate_shape(target_shape),
                    activation="relu",
                ),
                tf.keras.layers.Reshape(
                    (
                        self.caclulate_shape(target_shape),
                        self.caclulate_shape(target_shape),
                        128,
                    )
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    self.target_shape[2], kernel_size=5, strides=2, padding="same"
                ),
            ]
        )

    def caclulate_shape(self, target_shape):
        return int(tf.math.ceil(target_shape[0] / 2**3))
