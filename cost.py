import numpy as np
from operation import Operation


class cost(Operation):
    """
    Represents a cost function used in neural networks.

    Attributes:
        None

    Methods:
        forward(pred, y): Calculates the forward pass of the cost function.
        backward(pred, y): Calculates the backward pass of the cost function.

    """

    @staticmethod
    def forward(pred, y):
        """
        Calculates the forward pass of the cost function.

        Args:
            pred (numpy.ndarray): The predicted values.
            y (numpy.ndarray): The actual values.

        Returns:
            float: The cost value.

        """
        return np.mean(np.power(pred - y, 2))

    @staticmethod
    def backward(pred, y):
        """
        Calculates the backward pass of the cost function.

        Args:
            pred (numpy.ndarray): The predicted values.
            y (numpy.ndarray): The actual values.

        Returns:
            numpy.ndarray: The gradients of the cost function.

        """
        return 2 * (pred - y)
