#!/usr/bin/env python
import math

import numpy as np
from sklearn.model_selection import train_test_split


def f(x):
    return np.sin(x / 5)

np.random.seed(42)
n_samples = 200

X = np.random.uniform(-50, 50, n_samples)
Y = f(X) + np.random.randn(n_samples) * 0.25
X = X.reshape(-1, 1) # Convertim X en una matriu de 1 columna

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, Y_test = zip(*sorted(zip(X_test, Y_test))) # Ordenem les dades de test per visualitzar-les correctament


from sklearn.tree import DecisionTreeRegressor

max_depth = 5
model = DecisionTreeRegressor(max_depth=max_depth)

model.fit(X_train, Y_train)

pred_Y = model.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rmse = mean_squared_error(Y_test, pred_Y)
r2 = r2_score(Y_test, pred_Y)
print(f'RMSE tree: {rmse:.2f}')
print(f'R^2 tree: {r2:.2f}')

import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='orange')
plt.plot(X_test, pred_Y, color='red', lw=2)
plt.show()