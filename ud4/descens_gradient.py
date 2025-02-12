#!/usr/bin/env python
#!/usr/bin/env python
import numpy as np
from sklearn.model_selection import train_test_split

def f(x):
    return 3*x + 100

np.random.seed(42)
n_samples = 50

X = np.random.uniform(-50, 50, n_samples)
Y = f(X) + np.random.randn(n_samples) * 10
X = X.reshape(-1, 1) # Convertim X en una matriu de 1 columna
Y[10] = 1000

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y.reshape(-1, 1)).ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, Y_test = zip(*sorted(zip(X_test, Y_test))) # Ordenem les dades de test per visualitzar-les correctament


from sklearn.linear_model import SGDRegressor

max_iter = 1000
models = {
    "huber_small_e": SGDRegressor(loss='huber', max_iter=max_iter, epsilon=0.1),
    "huber_large_e": SGDRegressor(loss='huber', max_iter=max_iter, epsilon=2.5),
    "epsilon_insensitive": SGDRegressor(loss='epsilon_insensitive', max_iter=max_iter, epsilon=0.1),
    "squared_error": SGDRegressor(loss='squared_error', max_iter=max_iter),
}
colors = ['red', 'green', 'cyan', 'magenta']

import matplotlib.pyplot as plt

plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='orange')

for i, (title, model) in enumerate(models.items()):
    model.fit(X_train, Y_train)

    pred_Y = model.predict(X_test)

    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import r2_score

    rmse = root_mean_squared_error(Y_test, pred_Y)
    r2 = r2_score(Y_test, pred_Y)
    print(f'RMSE {title}: {rmse:.2f}')
    print(f'R^2 {title}: {r2:.2f}')

    X_plot = [X.min(), X.max()]
    Y_plot = model.predict(np.array(X_plot).reshape(-1, 1))

    plt.plot(X_plot, Y_plot, color=colors[i], lw=2)

plt.legend(["train", "test"] + list(models.keys()))
plt.show()