from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.metrics import mean_absolute_error as mae
import copy

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# regressor = Lasso(random_state=42,alpha=0.1)
regressor = SGDRegressor(random_state=42)

min_value = np.inf
X_new_feature = []
X_best_feature = []
for i in range(len(boston.feature_names)):
	for feature in boston.feature_names:
		if feature in X_new_feature:
			continue
		X_new_feature.append(feature)
		regressor.fit(X[X_new_feature], y)
		y_pred = regressor.predict(X[X_new_feature])
		mae_value = mae(y, y_pred)
		if mae_value < min_value:
			min_value = mae_value
			X_best_feature = copy.deepcopy(X_new_feature)
		X_new_feature.pop()  ##-- delete last

	X_new_feature = copy.deepcopy(X_best_feature)


X_best_feature_del = []
X_features = copy.deepcopy(X_new_feature)
for feature in X_new_feature:
	X_features.remove(feature)

	regressor.fit(X[X_features], y)
	y_pred = regressor.predict(X[X_features])
	mae_value = mae(y, y_pred)
	if mae_value < min_value:
		min_value = mae_value
		X_best_feature_del = copy.deepcopy(X_features)

	X_features = copy.deepcopy(X_new_feature)
