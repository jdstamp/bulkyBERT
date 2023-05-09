import h5py
import numpy as np
import random
import tensorflow as tf


def load_periodic_sims(sample_proportion=None):
    file_path = "../data/data_simulated/sim_periodic_signal.h5"
    data_group = "expression/data"
    labels_group = "expression/labels"
    return load_hdf5_data(file_path, data_group, labels_group, sample_proportion)


def load_varoquaux_data(sample_proportion=None):
    file_path = "../data/varoquaux.h5"
    data_group = "expression/data"
    labels_group = "expression/labels"
    return load_hdf5_data(file_path, data_group, labels_group, sample_proportion)


def load_hdf5_data(file_path, data_group, labels_group, sample_proportion):
    data_sim = h5py.File(file_path, "r")
    data = data_sim.get(data_group)
    labels = data_sim.get(labels_group)
    data = np.transpose(data[:], (1, 0))
    if len(labels.shape) > 1:
        labels = np.transpose(labels[:], (1, 0))
    else:
        labels = labels[:]
    if sample_proportion:
        sample_proportion = max(0, min(sample_proportion, 1))
        idx = random.sample(
            range(data.shape[0]), int(data.shape[0] * sample_proportion)
        )
        data = tf.gather(data, indices=idx)
        labels = tf.gather(labels, indices=idx)
    else:
        data = tf.gather(data, indices=list(range(data.shape[0])))
        labels = tf.gather(labels, indices=list(range(data.shape[0])))
    return data, labels


if __name__ == "__main__":
    pass
