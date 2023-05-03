import tensorflow as tf


class SoftLabels(tf.keras.layers.Layer):

    def __init__(self, num_clusters):
        super(SoftLabels, self).__init__()
        self.num_clusters = num_clusters

    def build(self, input_shape):
        self.soft_labels = self.add_weight(name="softlabels",
                                           shape=[self.num_clusters,
                                                  int(input_shape[-1])],
                                           initializer="glorot_uniform")

    def call(self, inputs):
        return self.students_t_pdf(inputs)

    def students_t_pdf(self, inputs, nu=1):
        t_squared = tf.keras.backend.sum(tf.keras.backend.square(
            tf.keras.backend.expand_dims(inputs, axis=1) - self.soft_labels
        ), axis=2)
        pdf = (1 + t_squared / nu)**(-(nu + 1) / 2)
        output = tf.transpose(tf.transpose(pdf) / tf.keras.backend.sum(pdf, axis=1))
        return output
