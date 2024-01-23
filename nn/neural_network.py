import numpy as np

from .layer import Layer
from .cost import MeanSquaredCost, CrossEntropyCost
from .data import LayerLearnData, NetworkLearnData


class NeuralNetwork:
    """
    Represents a neural network with multiple layers.

    Args:
        layer_sizes (list[int]): A list of integers representing the number of nodes in each layer.

    Attributes:
        layers (list[Layer]): A list of Layer objects representing the layers of the neural network.
        cost: The cost function used for training the neural network.

    Methods:
        calculate_outputs(inputs): Calculates the outputs of the neural network given the inputs.
        classify(inputs): Classifies the inputs using the neural network.
        update_gradients(data, expected, learn_data): Updates the gradients of the neural network during training.
        learn(x, y, learning_rate=0.1, epochs=100): Trains the neural network using the given inputs and expected outputs.
    """

    def __init__(self, layer_sizes, cost=MeanSquaredCost) -> None:
        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
        self.cost = cost

    def calculate_outputs(self, inputs):
        """
        Calculates the outputs of the neural network given the inputs.

        Args:
            inputs: The input values to the neural network.

        Returns:
            The output values of the neural network.
        """
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs[0] if len(inputs) == 1 else inputs

    def classify(self, inputs):
        """
        Classifies the inputs using the neural network.

        Args:
            inputs: The input values to classify.

        Returns:
            The maximum output value and the output values of the neural network.
        """
        r = self.calculate_outputs(inputs)
        return r.max(), r

    def update_gradients(self, data, expected, learn_data: NetworkLearnData):
        """
        Updates the gradients of the neural network during training.

        Args:
            data: The input data for training.
            expected: The expected output for the input data.
            learn_data: The learning data for the neural network.

        Returns:
            None
        """
        inputs = data
        for i, layer in enumerate(self.layers):
            inputs = layer.calculate_outputs(inputs, learn_data.layer_data[i])
        output_layer_index = len(self.layers) - 1
        output_layer: Layer = self.layers[output_layer_index]
        output_layer_data: LayerLearnData = learn_data.layer_data[output_layer_index]
        output_layer.calculate_output_layer_node_derivatives(
            output_layer_data, expected, self.cost
        )
        output_layer.update_gradients(output_layer_data)
        for i in range(output_layer_index - 1, -1, -1):
            layer_learn_data: LayerLearnData = learn_data.layer_data[i]
            hidden_layer: Layer = self.layers[i]
            hidden_layer.calculate_hidden_layer_node_derivatives(
                layer_learn_data,
                self.layers[i + 1],
                learn_data.layer_data[i + 1].node_values,
            )
            hidden_layer.update_gradients(layer_learn_data)

    def learn(self, x, y, learning_rate=0.1, epochs=100):
        """
        Trains the neural network using the given inputs and expected outputs.

        Args:
            x: The input data for training.
            y: The expected outputs for the input data.
            learning_rate (float): The learning rate for training the neural network. Default is 0.1.
            epochs (int): The number of training epochs. Default is 100.

        Returns:
            None
        """
        for _ in range(epochs):
            for i in range(len(x)):
                learn_data = NetworkLearnData(self.layers)
                self.update_gradients(x[i], y[i], learn_data)
            for l in self.layers:
                l.apply_gradients(learning_rate=learning_rate)
            yped = [self.calculate_outputs(x[i]) for i in range(len(x))]
            print(
                "Cost: ",
                np.sum(self.cost.forward(yped[i], y[i]) for i in range(len(x))),
            )

    def evaluate(self, x, y):
        """
        Evaluates the performance of the neural network on the given inputs and expected outputs.

        Args:
            x: The input data for evaluation.
            y: The expected outputs for the input data.

        Returns:
            float: The accuracy of the neural network on the evaluation data.
        """
        correct = 0
        total = len(x)

        for i in range(total):
            outputs = self.calculate_outputs(x[i])
            predicted_class = np.argmax(outputs)
            expected_class = np.argmax(y[i])

            if predicted_class == expected_class:
                correct += 1

        accuracy = correct / total
        return f"Accuracy: {accuracy}"
