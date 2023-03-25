import numpy as np
from Neural import NeuralNetwork
# create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
nn = NeuralNetwork(2, 4, 1)

# create a training dataset with 4 examples
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# train the neural network for 1000 epochs
nn.train(X_train, y_train, 1000)

# predict output for new input data
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_pred = nn.predict(X_test)
print(y_pred)
