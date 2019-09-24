import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0],
					[0, 1],
					[1, 0],
					[1, 1]])
expected_outputs = np.array([[0],
							[1],
							[1],
							[0]])

class XOR_Net():
	def __init__(self):
		self.epochs = 10000
		self.learning_rate = 0.1
		self.n_input_nodes = 2
		self.n_hidden_nodes = 2
		self.n_output_nodes = 1
		self.n = 4

		np.random.seed(0)
		self.hidden_weights = np.random.uniform(-1, 1, (self.n_input_nodes, self.n_hidden_nodes))
		self.hidden_bias = np.random.random((1, self.n_hidden_nodes))
		self.output_weights = np.random.uniform(-1, 1, (self.n_hidden_nodes, self.n_output_nodes))
		self.output_bias = np.random.random((1, self.n_output_nodes))

		self.cost = np.zeros((self.epochs))

	def activation(self, x, backward=False):
		if backward: return x * (1 - x)
		return 1 / (1 + np.exp(-x))

	def forward(self, x, predict=False):
		self.a0 = x.reshape(1, x.shape[0]) # 1x2

		self.z1 = np.dot(self.a0, self.hidden_weights) + self.hidden_bias # 1x2 * 2x2 + 1x2 = 1x2
		self.a1 = self.activation(self.z1) # 1x2

		self.z2 = np.dot(self.a1, self.output_weights) + self.output_bias # 1x2 * 2x1 + 1x1 = 1x1
		self.a2 = self.activation(self.z2) # 1x1

		if predict: return self.a2

	def backward(self, y):
		self.dz2 = y - self.a2 # 1x1 - 1x1 = 1x1
		self.d_output = self.dz2 * self.activation(self.a2, backward=True) # 1x1 .* 1x1 = 1x1

		self.dz1 = np.dot(self.d_output, self.output_weights.T) # 1x1 * 1x2 = 1x2
		self.d_hidden = self.dz1 * self.activation(self.a1, backward=True) # 1x2 .* 1x2 = 1x2

		self.dw1 += np.dot(self.a0.T, self.d_hidden) # 2x1 * 1x2 = 2x2
		self.dw2 += np.dot(self.a1.T, self.d_output) # 2x1 * 1x1 = 2x1
		self.db1 += self.d_hidden # 1x2
		self.db2 += self.d_output # 1x1

	def train(self, inputs, expected_outputs):
		for epoch in range(self.epochs):
			self.dw1 = 0
			self.dw2 = 0
			self.db1 = 0
			self.db2 = 0

			for x in range(self.n):
				self.forward(inputs[x])
				self.backward(expected_outputs[x])
				self.cost[epoch] += (-(expected_outputs[x] * np.log(self.a2)) - ((1 - expected_outputs[x]) * np.log(1 - self.a2)))

			self.hidden_weights += self.learning_rate * self.dw1
			self.output_weights += self.learning_rate * self.dw2
			self.hidden_bias += self.learning_rate * self.db1
			self.output_bias += self.learning_rate * self.db2

			print("Epoch {:04d}/{:04d}, Cost {:.4f}".format(epoch, self.epochs, self.cost[epoch]))

def solution():
	xor_net = XOR_Net()
	xor_net.train(inputs, expected_outputs)
	for i in inputs:
		print("input:", i)
		print("predicted output: {:.4f}".format(xor_net.forward(i, predict=True)[0][0]))
	plt.plot(range(xor_net.epochs), xor_net.cost)
	plt.xlabel("Epoch")
	plt.ylabel("Cost")
	plt.show()

solution()
