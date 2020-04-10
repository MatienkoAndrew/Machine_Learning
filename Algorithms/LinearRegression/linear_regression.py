import numpy as np

np.random.seed(42)
##-- First method (по формуле)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Коэффициенты:")
print(theta_best)

X_new = [[0], [2]]
X_new_b = np.c_[np.ones((2, 1)), X_new]
print("Тестовый набор X")
print(X_new_b)

y_pred = theta_best[0] + theta_best[1] * X_new
print("Предсказанные значения")
print(y_pred)

print("\n\n\nMethod2:\n\n")
##-- Второй метод

### y = b0 + b1 * x
##-- b0 =  mean(y) - b1 * mean(x)
##-- b1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum((x(i) - mean(x))^2)

##-- Calculate the mean value of the list numbers
def mean(values):
	return sum(values) / float(len(values))

##-- (дисперсия)
##-- Calculate the variance of the list numbers
def variance(values, mean):
	return sum([(x - mean) ** 2 for x in values])


##-- Ковариация
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

##-- Подсчет коэффициентов
def coefficients(X, y):
	x, y = X, y
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

intercept, coef = coefficients(X, y)
print("intercept = ", intercept)
print("coef = ", coef)


##-- Функция RMSE (root mean square error)
from math import sqrt

def rmse(actual, predict):
	sum_error = 0.0
	for i in range(len(actual)):
		sum_error += (predict[i] - actual[i]) ** 2
	return sqrt(sum_error / float(len(actual)))