import numpy as np
from .operation import Operation


class MeanSquaredCost(Operation):
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


class CrossEntropyCost(Operation):
    @staticmethod
    def forward(pred, y):
        """
        Compute the Cross-Entropy Cost.

        Parameters:
        - y_true: numpy array, true labels (ground truth)
        - y_pred: numpy array, predicted probabilities

        Returns:
        - cost: float, Cross-Entropy Cost
        """
        m = y.shape[0]  # number of training examples
        epsilon = 1e-15  # small constant to avoid log(0)

        # Clip values to avoid log(0) and log(1)
        y_pred = np.clip(pred, epsilon, 1 - epsilon)

        # Compute Cross-Entropy Cost
        cost = -(1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        return cost

    @staticmethod
    def backward(pred, y):
        """
        Compute the derivative of Cross-Entropy Cost with respect to y_pred.

        Parameters:
        - y_true: numpy array, true labels (ground truth)
        - y_pred: numpy array, predicted probabilities

        Returns:
        - derivative: numpy array, derivative of Cross-Entropy Cost with respect to y_pred
        """
        epsilon = 1e-15  # small constant to avoid division by zero

        # Clip values to avoid division by zero
        y_pred = np.clip(pred, epsilon, 1 - epsilon)

        # Compute the derivative
        derivative = -(y / y_pred) + (1 - y) / (1 - y_pred)

        return derivative


class CategorialCrossEntropyCost(Operation):
    @staticmethod
    def forward(pred, y):
        """
        Compute the Cross-Entropy Cost.

        Parameters:
        - y_true: numpy array, true labels (ground truth)
        - y_pred: numpy array, predicted probabilities

        Returns:
        - cost: float, Cross-Entropy Cost
        """
        return -np.sum(y * np.log(pred + 1e-15))

    @staticmethod
    def backward(pred, y):
        """
        Compute the derivative of Cross-Entropy Cost with respect to y_pred.

        Parameters:
        - y_true: numpy array, true labels (ground truth)
        - y_pred: numpy array, predicted probabilities

        Returns:
        - derivative: numpy array, derivative of Cross-Entropy Cost with respect to y_pred
        """
        c = []
        for i in range(len(pred)):
            if pred[i] in [0, 1]:
                c.append(0)
            else:
                c.append((-pred[i] + y[i]) / (pred[i] * (pred[i] - 1)))
        return np.array(c)
