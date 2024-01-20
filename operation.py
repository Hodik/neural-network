from abc import ABCMeta, abstractstaticmethod


class Operation(metaclass=ABCMeta):
    """
    Abstract base class for operations.

    Attributes:
        None

    Methods:
        forward(x): Performs the forward pass of the operation.
        backward(x): Performs the backward pass of the operation.
    """

    @abstractstaticmethod
    def forward(x):
        pass

    @abstractstaticmethod
    def backward(x):
        pass
