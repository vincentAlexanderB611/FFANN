import numpy as np
import pandas as pd

# Given values (smiling matrix)
a = [1.0, 0.5, 1.0]
b = [0.5, 1.0, 0.34]
c = [0.1, 1.0, 0.23]
df = {1: a, 2: b, 3: c}
matrix = np.array(pd.DataFrame(df))

d = [0.93, 0.23, 0.12]
e = [0.22, 0.76, 0.32]
f = [0.20, 0.34, 0.23]
wght = {1: d, 2: e, 3:f}
weights = np.array(pd.DataFrame(wght))


class X_input:
    def __init__(self, value):
        self.value = value

    def matrix_summation(m, w):
        return (np.sum(m * w))

class Neuron(X_input):
    def __init__(self, value, w, b):
        super().__init__(value)
        self.w = w
        self.b = b

    def relu(self):
        return max(0, self.value * self.w + self.b)

    def sigmoid(self):
        return 1 / (1 + np.exp(-self.value * self.w - self.b))

input_value = X_input.matrix_summation(matrix, weights)

# Creating X_input objects
x1 = X_input(input_value)
x2 = X_input(input_value)
x3 = X_input(input_value)

# Creating Neuron objects and applying ReLU activation
l11 = Neuron(input_value, 1.4, 0).relu()
l12 = Neuron(input_value, 0.8, 1).relu()
l13 = Neuron(input_value, 0.5, 3).relu()

first_layer_sum = np.sum(np.array([l11, l12, l13]))

# first hidden layer
h11 = Neuron(first_layer_sum, 1.5, 1.0).relu()
h12 = Neuron(first_layer_sum, 0.8, 2.0).relu()
h13 = Neuron(first_layer_sum, 0.5, 3.0).relu()

hid_layer = np.array([h11, h12, h13])
hid_layer_sum = np.sum(hid_layer)

# Output layer needs to receive weight * sum summation
l21 = Neuron(hid_layer_sum, 0.5, 1).sigmoid()
l22 = Neuron(hid_layer_sum, 0.1, 1).sigmoid()

print(f'outputs:\n{l21, l22}')

