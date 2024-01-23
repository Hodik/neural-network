import numpy as np

from .data import LayerLearnData
from .activation import relu, sigmoid, tanh
from .operation import Operation
from typing import Self


class Layer:
    def __init__(self, n_in, n_out, activation: Operation | None = None):
        """
        Initializes a Layer object.

        Args:
            n_in (int): Number of input nodes.
            n_out (int): Number of output nodes.
            activation (Operation, optional): Activation function to be used. Defaults to None.
        """
        self.activation = activation or sigmoid
        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.biases = np.random.randn(n_out) * np.sqrt(2 / n_in)
        self.gradW = np.zeros(self.weights.shape)
        self.gradB = np.zeros(self.biases.shape)

    def set_activation(self, activation: Operation):
        """
        Sets the activation function of the layer.

        Args:
            activation (Operation): Activation function to be set.
        """
        self.activation = activation

    def calculate_outputs(self, inputs, learn_data: LayerLearnData | None = None):
        """
        Calculates the outputs of the layer given the inputs.

        Args:
            inputs: Input values.
            learn_data (LayerLearnData, optional): Learning data object. Defaults to None.

        Returns:
            np.ndarray: Array of output values.
        """
        output = []
        for i, w in enumerate(self.weights.T):
            output.append(np.dot(w, inputs) + self.biases[i])

        if learn_data:
            learn_data.inputs = np.array(inputs)
            learn_data.weighted_inputs = np.array(output)
            learn_data.activations = self.activation.forward(np.array(output))

        return self.activation.forward(np.array(output))

    def calculate_output_layer_node_derivatives(
        self, learn_data: LayerLearnData, expected, cost: Operation
    ):
        """
        Calculates the derivatives of the output layer nodes.

        Args:
            learn_data (LayerLearnData): Learning data object.
            expected: Expected output values.
            cost (Operation): Cost function.

        Returns:
            None
        """
        cost_derivative = cost.backward(learn_data.activations, expected)
        activation_derivative = self.activation.backward(learn_data.weighted_inputs)
        learn_data.node_values = cost_derivative * activation_derivative

    def calculate_hidden_layer_node_derivatives(
        self, learn_data: LayerLearnData, next_layer: Self, next_layer_values
    ):
        """
        Calculates the derivatives of the hidden layer nodes.

        Args:
            learn_data (LayerLearnData): Learning data object.
            next_layer (Layer): Next layer object.
            next_layer_values: Values of the next layer nodes.

        Returns:
            None
        """
        for i in range(len(learn_data.node_values)):
            v = 0
            for ni in range(len(next_layer_values)):
                v += next_layer.weights[i][ni] * next_layer_values[ni]
            v *= self.activation.backward(learn_data.weighted_inputs)[i]

            learn_data.node_values[i] = v

    def update_gradients(self, learn_data: LayerLearnData):
        """
        Updates the gradients of the layer.

        Args:
            learn_data (LayerLearnData): Learning data object.

        Returns:
            None
        """
        for out in range(self.n_out):
            node_derivative = learn_data.node_values[out]
            for inp in range(self.n_in):
                self.gradW[inp][out] += node_derivative * learn_data.inputs[inp]

            self.gradB[out] += node_derivative

    def apply_gradients(self, learning_rate):
        """
        Applies the gradients to the weights and biases of the layer.

        Args:
            learning_rate: Learning rate.

        Returns:
            None
        """
        self.weights -= learning_rate * self.gradW
        self.biases -= learning_rate * self.gradB

        self.gradW = np.zeros(self.weights.shape)
        self.gradB = np.zeros(self.biases.shape)

    def __repr__(self) -> str:
        return f"Layer(n_in={self.n_in}, n_out={self.n_out}, activation={self.activation}, weights={self.weights}, biases={self.biases}, gradW={self.gradW}, gradB={self.gradB})"
