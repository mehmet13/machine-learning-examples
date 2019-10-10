"""
Homework1 by Mehmet Emin Yıldırım
"""

import matplotlib.pyplot as plt
import ysa.activation_functions as acf


class Homework():
	"""
	Homework class is written for artificial neural networks lecture's 1st homework
	"""
    def __init__(self):
        self.x = acf.np.arange(-5.0, 5.0, 0.1)
        self.plot()
        self.set_options()
        self.show()

    def plot(self):
		"""
		The method that plots activation functions			
		"""
        plt.plot(self.x, acf.sigmoid(self.x), label='sigmoid')
        plt.plot(self.x, acf.linear(self.x, 0.1), label='linear (c = 0.1)')
        plt.plot(self.x, acf.tanh(self.x), label='tanh')
        plt.plot(self.x, acf.relu(self.x), label='relu')
        plt.plot(self.x, acf.leaky_relu(self.x), label='leaky relu')
        plt.plot(self.x, acf.softmax(self.x), label='softmax')
        plt.plot(self.x, acf.swish(self.x), label='swish (β = 1)')

    def set_options():
		"""
		The method that sets options
		"""
        plt.title('Homework 1')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.gca().set_ylim(-1.1, 2)

    def show():
		"""
		The method that shows the graph
		"""
        plt.show()

# Run Homework
homework = Homework()
