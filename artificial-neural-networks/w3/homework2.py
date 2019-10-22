"""
Perceptron Class
"""
import numpy as np


class Perceptron(object):
    def __init__(self, weights, learning_rate=0.5, bias=0):
        """
        Constructor of Perceptron class.

        Args:
            weights(ndarray): The weights of perceptron.
            learning_rate(float): The learning rate of perceptron.
            bias(float): The bias value.
        """
        self.bias = bias
        self.weights = weights
        self.learning_rate = learning_rate

    def train(self, inputs, desire):
        """
        The method that uses inputs and desire arguments to train the weights.

        Args:
            inputs(ndarray): The input array for training.
            desire(ndarray): The expected values according to inputs.

        Usage:
            x = np.array([1, 0], [0, 1])
            d = np.array([1, 0])
            p.train(x, d)
        """
        self.learning_ended = True
        for i in range(len(inputs)):
            fnet = self.find_fnet(inputs[i])
            err = desire[i] - fnet
            if err != 0:
                self.learning_ended = False
                self.update_weights(err, inputs[i])
        if not self.learning_ended:
            self.train(inputs, desire)

    def activation_function(self, net):
        """
        The activation function used by Perceptron class.

        Args:
            net(float): The net value calculated by find_fnet() method.

        Returns:
            The return value. Returns 1 or 0 according to value of net.
        """
        if net > 0:
            return 1
        else:
            return 0

    def find_fnet(self, inpt):
        """
        The method that calculates net and returns fnet

        Args:
            inpt(ndarray): The input row for calculating net with dot product
                           and bias.

        Returns:
            Returns fnet with using activation function
        """
        net = np.dot(inpt, self.weights) + self.bias
        return self.activation_function(net)

    def update_weights(self, err, inpt):
        """
        The method that updates weights of Perceptron after error occurs.

        Args:
            err(float): The error value.
            inpt(ndarray): The input row where error occurred.
        """
        self.weights = self.weights + self.learning_rate * err * inpt

    def predict(self, inputs):
        """
        The method that predicts and prints output according to inputs
        """
        result = np.zeros(len(inputs))
        for i in range(len(inputs)):
            fnet = self.find_fnet(inputs[i])
            result[i] = fnet
        print(result)
