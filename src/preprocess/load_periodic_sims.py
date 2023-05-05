import h5py
import numpy as np
import random
import tensorflow as tf


def load_periodic_sims(sample_proportion=None):
    data_sim = h5py.File("../data/data_simulated/sim_periodic_signal.h5", "r")
    periodic_signal = data_sim.get("expression/data")
    periodic_labels = data_sim.get("expression/labels")
    periodic_signal = np.transpose(periodic_signal[:], (1, 0))
    periodic_labels = periodic_labels[:]
    if sample_proportion:
        sample_proportion = max(0, min(sample_proportion, 1))
        idx = random.sample(
            range(len(periodic_labels)), int(len(periodic_labels) * sample_proportion)
        )
        periodic_signal = tf.gather(periodic_signal, indices=idx)
        periodic_labels = tf.gather(periodic_labels, indices=idx)
    return periodic_signal, periodic_labels
