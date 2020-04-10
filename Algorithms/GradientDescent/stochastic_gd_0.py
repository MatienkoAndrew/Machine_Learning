import numpy as np
np.random.seed(42)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

def predict(row, coefficients):
	y_pred = coefficients[0]
	for i in range(len(row) - 1):
		y_pred += coefficients[i + 1] * row[i]
	return y_pred

def sgd(train, l_rate=0.001, n_epochs=50):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epochs):
		sum_error = 0.0
		for row in train:
			y_pred = predict(row, coef)
			error = y_pred - row[-1]
			sum_error += error ** 2
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row) - 1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
	return coef

coefs = sgd(list(zip(X, y)))
print(coefs) # [array([8.23381037]), array([-0.00918622])]