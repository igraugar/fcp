import numpy as np
import tensorflow as tf


def forward_composition(estimator, x):
    """
    Compute compositions with the FCP method for a given instance.

    Parameters
    ----------
    estimator : keras.models.Model
        A trained neural network model.
    x : array-like
        The instance to be explained, with a length of n_features.

    Returns
    -------
    compositions : list of np.ndarray
        List of matrices representing compositions for each layer.
    """

    Theta = np.identity(len(x))
    compositions = [Theta]  # list of matrices \theta

    tensor_A = np.reshape(x, (-1, x.shape[0]))

    for l in range(1, len(estimator.layers)):

        tensor_A = estimator.layers[l - 1](tensor_A)

        A = np.absolute(tf.keras.backend.eval(tensor_A).transpose().flatten())  # activation vector

        W = tf.keras.backend.eval(estimator.layers[l].get_weights()[0])  # get_weights() [0] weights [1] biases

        R = (Theta.transpose() * A)  # transpose is needed for braodcasting element-wise multiplication

        Theta_temp = np.matmul(R, W).transpose()
        Theta = (Theta_temp.transpose() / np.absolute(Theta_temp).sum(axis=1)).transpose()  # normalizing

        compositions.append(Theta)

    return compositions

