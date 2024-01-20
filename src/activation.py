import numpy as np
from .operation import Operation


class sigmoid(Operation):
    """
    Implements the sigmoid activation function.

    Args:
        x (float): Input value.

    Returns:
        float: Output value after applying the sigmoid function.

    Raises:
        None

    Examples:
        >>> sigmoid.forward(0)
        0.5
        >>> sigmoid.backward(0)
        0.25
    """

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        return sigmoid.forward(x) * (1 - sigmoid.forward(x))


class softmax(Operation):
    """
    Computes the softmax activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying softmax function.

    Raises:
        None

    Examples:
        >>> x = np.array([1, 2, 3])
        >>> softmax.forward(x)
        array([0.09003057, 0.24472847, 0.66524096])
    """

    @staticmethod
    def forward(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def backward(x):
        return softmax.forward(x) * (1 - softmax.forward(x))


class relu(Operation):
    """
    Implements the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying ReLU activation.

    Examples:
        >>> x = np.array([-1, 0, 1])
        >>> relu.forward(x)
        array([0, 0, 1])

    References:
        - ReLU Activation Function: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """

    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(x):
        return np.where(x <= 0, 0, 1)
