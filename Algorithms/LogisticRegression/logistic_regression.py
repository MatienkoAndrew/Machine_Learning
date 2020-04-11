import numpy as np
from sklearn.datasets import load_iris
np.random.seed(42)

class LogisticRegression:
	def __init__(self,
	             learning_rate=0.01,
	             n_iter=100000,
	             fit_intercept=True,
	             verbose=False):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.fit_intercept = fit_intercept
		self.verbose = verbose

	def __add_intercept(self, X):
		intercept = np.ones((X.shape[0], 1))
		return np.concatenate((intercept, X), axis=1)

	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def __loss(self, h, y):
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

	def fit(self, X, y):
		if self.fit_intercept:
			X = self.__add_intercept(X)

		##weights init
		self.theta = np.zeros(X.shape[1])

		for i in range(self.n_iter):
			z = np.dot(X, self.theta)
			h = self.__sigmoid(z)
			gradient = np.dot(X.T, (h - y)) / y.size
			self.theta -= self.learning_rate * gradient

			if self.verbose == True and i % 10000 == 0:
				z = np.dot(X, self.theta)
				h = self.__sigmoid(z)
				print(f'loss: {self.__loss(h, y)} \t')

	def predict_proba(self, X):
		if self.fit_intercept:
			X = self.__add_intercept(X)

		return self.__sigmoid(np.dot(X, self.theta))

	def predict(self, X, threshold):
		return self.predict_proba(X) >= threshold


iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.rand(1, 100)


clf = LogisticRegression(learning_rate=0.1, n_iter=300000)
# clf.__add_intercept(X)
clf.fit(X, y)
y_pred = clf.predict_proba(X)
print(y_pred)
# print("Accuracy:", (y_pred==y).mean())