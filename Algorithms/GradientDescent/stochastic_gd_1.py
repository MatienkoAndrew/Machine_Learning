import numpy as np
np.random.seed(42)

def learning_schedule(t, t0=5, t1=50):
	return t0 / (t + t1)


def stochastic_gradient_descent(X, y, n_epochs=50):
	X_b = np.c_[np.ones((100, 1)), X]
	m = len(X)

	theta = np.random.randn(2, 1) ## Random init
	for epoch in range(n_epochs):
		for i in range(m):
			random_index = np.random.randint(m)
			xi = X_b[random_index : random_index + 1]
			yi = y[random_index : random_index + 1]
			gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
			eta = learning_schedule(epoch * m + i)
			theta = theta - eta * gradient
	return theta

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
b0, b1 = stochastic_gradient_descent(X, y)
print(b0, b1) # [4.19000137] [2.73381258]