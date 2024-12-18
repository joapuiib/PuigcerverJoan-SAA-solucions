#!/usr/bin/env python
import math

import numpy as np
from sklearn.model_selection import train_test_split


def f(x):
    # return -100 - 5*x + 50*np.power(x, 2) + 10*np.power(x, 3) + -0.2*np.power(x, 4)
    return np.sin(x / 5)

np.random.seed(42)
n_samples = 200

X = np.random.uniform(-50, 50, n_samples)
Y = f(X) + np.random.randn(n_samples) * 0.25
X = X.reshape(-1, 1) # Convertim X en una matriu de 1 columna

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, Y_test = zip(*sorted(zip(X_test, Y_test))) # Ordenem les dades de test per visualitzar-les correctament

import matplotlib.pyplot as plt

plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='orange')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train, Y_train)

pred_Y = model.predict(X_test)

rmse = mean_squared_error(Y_test, pred_Y)
r2 = r2_score(Y_test, pred_Y)
print(f'RMSE linear: {rmse:.2f}')
print(f'R^2 linear: {r2:.2f}')

plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='orange')
plt.plot(X_test, pred_Y, color='red')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

colors = ['red', 'cyan', 'purple', 'green', 'yellowgreen']

plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='green')
for degree in range(1, 6):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    model = make_pipeline(polynomial_features, LinearRegression())

    model.fit(X_train, Y_train)

    print(f'Coeficients: {model.named_steps["linearregression"].coef_}')
    print(f'Intercept: {model.named_steps["linearregression"].intercept_}')

    pred_Y = model.predict(X_test)

    rmse = mean_squared_error(Y_test, pred_Y)
    r2 = r2_score(Y_test, pred_Y)
    print(f'RMSE polinòmica: {rmse:.2f}')
    print(f'R^2 polinòmica: {r2:.2f}')

    plt.plot(X_test, pred_Y, color=colors[degree-2], label=f'Grau {degree}', lw=2)
plt.show()