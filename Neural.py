import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize weights randomly with mean 0 and standard deviation 1
        self.weights_input_hidden = np.random.normal(0, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.normal(0, 1, (hidden_size, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # calculate dot product of input and hidden layer weights
        hidden_layer = np.dot(X, self.weights_input_hidden)
        # apply sigmoid activation function to hidden layer
        hidden_layer_activation = self.sigmoid(hidden_layer)
        # calculate dot product of hidden layer and output layer weights
        output_layer = np.dot(hidden_layer_activation, self.weights_hidden_output)
        # apply sigmoid activation function to output layer
        output_layer_activation = self.sigmoid(output_layer)
        hidden_layer_activation =elf.hidden_layer_activation = None  # define instance variable to store hidden layer activation
        return output_layer_activation
    
    def backward(self, X, y, output):
        # calculate error between predicted and actual output
        output_error = y - output
        # calculate derivative of output layer activation function
        output_derivative = self.sigmoid_derivative(output)
        # calculate error for hidden layer using chain rule
        hidden_error = np.dot(output_error * output_derivative, self.weights_hidden_output.T)
        # calculate derivative of hidden layer activation function
        hidden_derivative = self.sigmoid_derivative(hidden_layer_activation)
        # update weights using gradient descent
        self.weights_hidden_output += np.dot(hidden_layer_activation.T, output_error * output_derivative)
        self.weights_input_hidden += np.dot(X.T, hidden_error * hidden_derivative)
        
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
    def predict(self, X):
        return self.forward(X)
