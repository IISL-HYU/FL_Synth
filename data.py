import numpy as np
import tensorflow as tf

def MNIST(_type, batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test / 255.0
    x_syn = np.zeros(x_train.shape)

    return (x_train, y_train), (x_test, y_test), (x_syn, y_train)