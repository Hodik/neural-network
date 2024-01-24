import numpy as np
from nn.neural_network import NeuralNetwork
from nn.activation import softmax, sigmoid, relu
from nn.cost import CategorialCrossEntropyCost
from nn.helpers.one_hot import to_one_hot, from_one_hot


n = NeuralNetwork([728, 20, 20, 10], cost=CategorialCrossEntropyCost)
xs = np.random.rand(4, 728)
ys = [to_one_hot(np.random.randint(0, 10), 10) for _ in range(4)]
x = np.array(xs)
y = np.array(ys)
n.layers[0].set_activation(relu)
n.layers[1].set_activation(relu)
n.layers[2].set_activation(softmax)
n.learn(x, y, learning_rate=0.01, epochs=100)

print("Pred", from_one_hot(n.classify(x[0])[1]), "expected", y[0])

n.save_model("model.pkl")


new_model = NeuralNetwork.load_model("model.pkl")
print("Pred", from_one_hot(new_model.classify(x[0])[1]), "expected", y[0])
