import numpy as np


def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def from_one_hot(y):
    return np.argmax(y)
