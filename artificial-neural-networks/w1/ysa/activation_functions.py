"""
Activation Functions Module
"""
import numpy as np


def linear(x, c):
    """
    Returns result array of linear activation function according to given x
    and c arguments.

    :Args:
        x: x values array
        c: constant value

    :Usage:
        acf.linear(x, 1)
    """
    return x * c


def sigmoid(x):
    """
    Returns result array of sigmoid activation function according to given x
    argument.

    :Args:
        x: x values array

    :Usage:
        acf.sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Returns result array of tanh activation function according to given x
    argument.

    :Args:
        x: x values array

    :Usage:
        acf.tanh(x)
    """
    return 2 / (1 + np.exp(-2 * x)) - 1


def relu(x):
    """
    Returns result array of ReLU activation function according to given x
    argument.

    :Args:
        x: x values array

    :Usage:
        acf.relu(x)
    """
    zero = np.zeros(len(x))
    return np.max([zero, x], axis=0)


def leaky_relu(x):
    """
    Returns result array of Leaky ReLU activation function according to given x
    argument.

    :Args:
        x: x values array

    :Usage:
        acf.leaky_relu(x)
    """
    return np.max([0.01 * x, x], axis=0)


def softmax(x):
    """
    Returns result array of softmax activation function according to given x
    argument.

    :Args:
        x: x values array

    :Usage:
        acf.softmax(x)
    """
    return np.exp(x) / sum(np.exp(x))


def swish(x, beta=1):
    """
    Returns result array of swish activation function according to given x
    and beta arguments.

    :Args:
        x: x values array
        beta: beta multiplier in swish formula

    :Usage:
        acf.swish(x, 1)
    """
    return x * sigmoid(x * beta)
