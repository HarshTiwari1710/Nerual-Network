
# Multi-Layer Neural Network

This is a Python implementation of a multi-layer neural network from scratch using the NumPy library. The network has one hidden layer and uses the sigmoid activation function. The purpose of this project is to demonstrate how a neural network can be built and trained using only basic linear algebra and calculus concepts.




## Dependencies

- Python 3.7

* Numpy
## Python Documentation

[Refer Here](https://docs.python.org/3.7/)


## Usage
To use this neural network, follow these steps:

1. Create an instance of the NeuralNetwork class with the appropriate input size, hidden size, and output size.

```python
nn = NeuralNetwork(input_size, hidden_size, output_size)
```
2. Train the neural network on a dataset using the train method. The X parameter should be a NumPy array of input examples, and the y parameter should be a NumPy array of corresponding output examples. The epochs parameter controls the number of iterations to train for.

```python
nn.train(X, y, epochs)
```
3. Use the trained neural network to predict the output for new input data using the predict method. The X parameter should be a NumPy array of input examples.

```python
y_pred = nn.predict(X)
```
To see an example usage of the neural network, run the example.py script:

```python
python example.py
```
This will create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron, and train it on the XOR function using a training dataset with 4 examples. The network is then used to predict the output for the same input data, and the predicted outputs are printed.
## Author

- This project was created by [@HarshTiwari1710](https://github.com/HarshTiwari1710)

