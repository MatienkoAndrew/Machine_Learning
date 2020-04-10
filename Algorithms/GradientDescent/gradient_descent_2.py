import numpy as np

np.random.seed(42)

def gradient_descent(X, y):
	X_b = np.c_[np.ones((100, 1)), X]
	eta = 0.1
	n_iterations = 1000
	m = float(len(X))

	theta = np.random.randn(2, 1) ## Random init
	for iteration in range(n_iterations):
		gradient = 2/m * X_b.T.dot(X_b.dot(theta) - y)
		theta = theta - eta * gradient
	return theta

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
b0, b1 = gradient_descent(X, y)
print(b0, b1) # [4.21509616] [2.77011339]