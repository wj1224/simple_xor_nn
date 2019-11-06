import numpy as np
import matplotlib.pyplot as plt

class XOR_Net():
	def __init__(self):
		self.epochs = 10000
		self.learning_rate = 0.1
		self.n_input_nodes = 2
		self.n_hidden_nodes = 2
		self.n_output_nodes = 1
		self.n = 4

		np.random.seed(0)
		self.hidden_weights = np.random.uniform(-1, 1, (self.n_input_nodes, self.n_hidden_nodes)) # 2x2
		self.hidden_bias = np.random.random((1, self.n_hidden_nodes)) # 1x2
		self.output_weights = np.random.uniform(-1, 1, (self.n_hidden_nodes, self.n_output_nodes)) # 2x1
		self.output_bias = np.random.random((1, self.n_output_nodes)) # 1x1

		self.cost = np.zeros((self.epochs))

	def activation(self, x, backward=False):
		if backward: return x * (1 - x)
		return 1 / (1 + np.exp(-x))

	def forward(self, x, predict=False):
		a0 = x.reshape(1, x.shape[0]) # 1x2

		z1 = np.dot(a0, self.hidden_weights) + self.hidden_bias # 1x2 * 2x2 + 1x2 = 1x2
		a1 = self.activation(z1) # 1x2

		z2 = np.dot(a1, self.output_weights) + self.output_bias # 1x2 * 2x1 + 1x1 = 1x1
		a2 = self.activation(z2) # 1x1

		if predict: return a2

		return a2, a1, a0

	def backward(self, layers_output, gradients, y):
		a2 = layers_output[0]
		a1 = layers_output[1]
		a0 = layers_output[2]
		dw1 = gradients[0]
		dw2 = gradients[1]
		db1 = gradients[2]
		db2 = gradients[3]

		dz2 = y - a2 # 1x1 - 1x1 = 1x1
		d_output = dz2 * self.activation(a2, backward=True) # 1x1 .* 1x1 = 1x1

		dz1 = np.dot(d_output, self.output_weights.T) # 1x1 * 1x2 = 1x2
		d_hidden = dz1 * self.activation(a1, backward=True) # 1x2 .* 1x2 = 1x2

		dw1 += np.dot(a0.T, d_hidden) # 2x1 * 1x2 = 2x2
		dw2 += np.dot(a1.T, d_output) # 2x1 * 1x1 = 2x1
		db1 += d_hidden # 1x2
		db2 += d_output # 1x1

		return dw1, dw2, db1, db2

	def train(self, inputs, expected_outputs):
		for epoch in range(1, self.epochs + 1):
			dw1 = 0
			dw2 = 0
			db1 = 0
			db2 = 0

			for x in range(self.n):
				a2, a1, a0 = self.forward(inputs[x])
				dw1, dw2, db1, db2 = self.backward([a2, a1, a0], [dw1, dw2, db1, db2], expected_outputs[x])
				self.cost[epoch - 1] += (-(expected_outputs[x] * np.log(a2)) - ((1 - expected_outputs[x]) * np.log(1 - a2)))

			self.hidden_weights += self.learning_rate * dw1
			self.output_weights += self.learning_rate * dw2
			self.hidden_bias += self.learning_rate * db1
			self.output_bias += self.learning_rate * db2

			print("Epoch {:04d}/{:04d}, Cost {:.4f}".format(epoch, self.epochs, self.cost[epoch - 1]))

	def predict(self, inputs):
		for i in inputs:
			print("input", i)
			print("predicted output: {:.4f}".format(self.forward(i, predict=True)[0][0]))
		plt.plot(range(xor_net.epochs), xor_net.cost)
		plt.xlabel("Epoch")
		plt.ylabel("Cost")
		plt.show()

if __name__ == "__main__":
	inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	expected_outputs = np.array([[0], [1], [1], [0]])

	xor_net = XOR_Net()
	xor_net.train(inputs, expected_outputs)
	xor_net.predict(inputs)
