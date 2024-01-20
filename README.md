# Neural Network from Scratch

This repository contains a simple implementation of a neural network built from scratch in Python. The neural network uses only basic Python libraries to demonstrate the underlying mechanisms of neural networks.

## Files

- `neural_network.py`: This file contains the `NeuralNetwork` class, which represents a neural network with multiple layers.
- `layer.py`: This file contains the `Layer` class, which represents a single layer in the neural network.
- `cost.py`: This file contains the cost function used for training the neural network.
- `data.py`: This file contains the `LayerLearnData` and `NetworkLearnData` classes, which are used for learning data in the network and its layers.

## Usage

To use this neural network, you need to create an instance of the `NeuralNetwork` class with a list of integers representing the number of nodes in each layer. Then, you can train the network using the `train` method and make predictions with the `predict` method.

## Example

```python
from neural_network import NeuralNetwork

# Create a neural network with 2 input nodes, 2 hidden nodes, and 1 output node
nn = NeuralNetwork([2, 2, 1])

# Train the network
nn.train(X_train, y_train)

# Make a prediction
prediction = nn.predict(X_test)
