import random
import numpy as np
from types import *
from typing import *
from typing_extensions import *


class NN(object):

    @staticmethod
    def sigmoid(x: int|float) -> float:
        return 1 / (1 + np.exp(-np.array(x)))
    
    @staticmethod
    def sigmoid_deriv(x: float|int) -> float:
        return NN.sigmoid(x) * (1 - NN.sigmoid(x))

    @staticmethod
    def relu(x: int|float) -> int|float:
        return max((0, x))
    
    @staticmethod
    def mse(x: float|int, y: float|int) -> float:
        return np.square(x - y)

    @staticmethod
    def mse_deriv(x: float|int, y: float|int) -> float:
        try:
            return 2 * np.square(x - y)
        except:
            return 2 * np.square(np.array([*x, 1.0]), y)

    def __init__(self, i: int, h: Sequence[int], o: int) -> NoneType:
        self.i = i
        self.h = h
        self.o = o

    def init(self, fn: FunctionType) -> NoneType:
        self.weights = []

        s = [self.i, *self.h, self.o]

        for i in range(len(s) - 1):
            self.weights.append(np.random.randn(s[i]+1, s[i+1]))

        self.biases = [0] * (len(s) - 1)

        self.gradients = []

        self.fn = fn
    
    def input(self, inputs: Sequence[int|float]):
        if len(inputs) != self.i:
            raise RuntimeError("size of inputs doesn't match")
        
        self.inputs = inputs.copy()
    
    def process(self):
        outputs = []

        self.layers = []

        inputs = self.inputs.copy()

        for index, weight in enumerate(self.weights):
            self.layers.append(inputs)

            inputs = np.array(list(map(self.fn, [*inputs, self.biases[index]] @ weight)))
        
        for inp in inputs:
            outputs.append(inp)
        
        self.outputs = outputs.copy()
    
    def output(self):
        return self.outputs.copy()
    
    def gradient(self, index, target=np.array([0])):
        index_old = index
        index = -index - 1

        if index == -1:
            nxt = np.array(self.outputs) - target
        else:
            nxt = self.layers[index + 1]
        
        if index == -len(self.layers):
            prv = np.array(self.inputs)
        else:
            prv = self.layers[index - 1]
        
        deriv_error = NN.mse_deriv(prv, nxt)

        layer = NN.sigmoid_deriv(prv)

        bias = self.biases[index]

        deriv_weights = nxt

        delta_bias = (
            deriv_error * layer * bias
        )

        delta_weights = (
            deriv_error * layer * deriv_weights
        )

        self.gradients.append((delta_bias, delta_weights))

        if index != -len(self.layers):
            self.gradient(index_old + 1)

    def total_gradients(self, target):

        self.gradients = []

        self.gradient(0, target)

        self.gradients.reverse()
    
    def update_params(self):
        for index in range(len(self.gradients) - 1):
            bias = self.biases[index] - self.gradients[index][0] * self.learning_rate
            weights = self.weights[index] - (self.gradients[index][1] * self.learning_rate)

            self.biases[index] = bias
            self.weights[index] = weights

    def train(self, target, learning_rate: int|float = 1) -> NoneType:
        self.learning_rate = learning_rate

        self.total_gradients(target)

        self.update_params()


if __name__ == '__main__':
    nn = NN(2, [3, 3, 3], 1)

    nn.init(NN.sigmoid)

    train_data = [
        {
            "input": [0, 0],
            "output": [0]
        },
        {
            "input": [0, 1],
            "output": [1]
        },
        {
            "input": [1, 0],
            "output": [1]
        },
        {
            "input": [1, 1],
            "output": [0]
        },
    ]

    tries = 100

    for _ in range(tries):
        print("Process:", _ * 100 // tries, "%")

        data = random.choice(train_data)

        nn.input(data["input"])

        nn.process()

        nn.train(data["output"])
    
    print("Done! Try it)\n")

    while True:
        try:
            inp = eval(input().replace(" ", ", "))

            nn.input(inp)

            nn.process()

            print(nn.output())
        except KeyboardInterrupt:
            break
