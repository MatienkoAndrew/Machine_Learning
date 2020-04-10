import numpy as np
from typing import Dict, Tuple

##-- Первая реализация (простейшая)
def gradient_descent(x_array: np.ndarray,
                     y_array: np.ndarray,
                     b0_0: float,
                     b1_0: float,
                     epochs: int,
                     learning_rate: float=0.001
                     ) -> Tuple[float, float]:
	b0, b1 = b0_0, b1_0
	k = float(len(x_array))
	for i in range(epochs):
		## Вычисление новых предсказанных значений
		y_pred = b0 + b1 * x_array

		dL_db1 = (-2/k) * np.sum(np.multiply(x_array, y_array - y_pred))
		dL_db0 = (-2/k) * np.sum(y_array - y_pred)

		b1 = b1 - learning_rate * dL_db1
		b0 = b0 - learning_rate * dL_db0

	y_pred = b0 + b1 * x_array
	return b0, b1, y_pred


np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
b0, b1, y_pred = gradient_descent(X, y, 0, 0, 100000)
print(b0, b1) # 4.215096157545669 2.770113386439396