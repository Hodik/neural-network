import numpy as np
from nn.neural_network import NeuralNetwork

n = NeuralNetwork([3, 4, 4, 1])
x = np.array([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
y = np.array([1, 0, 0, 1])

n.learn(x, y, learning_rate=1, epochs=500)


print("Pred", n.classify([2.0, 3.0, -1.0])[0])
