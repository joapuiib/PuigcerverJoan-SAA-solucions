import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) -3
y = 0.5 * X**2 + X + 2 +np.random.rand(m, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

polynomial_features = PolynomialFeatures(degree=2)
model = make_pipeline(polynomial_features, LinearRegression())
# model.fit(X_train, y_train)

# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)

# mse_train = mean_squared_error(y_train, y_train_pred)
# print("MSE entrenamiento: %.2f" % mse_train)
# mse_test = mean_squared_error(y_test, y_test_pred   )
# print("MSE test: %.2f" % mse_test)

# Generate 100 samples from the whole range of X
#X_model = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#y_model = model.predict(X_model)

# Plot training data and test data with different colors
# Plot the model in red in a line
# plt.scatter(X_train, y_train, color='blue')
# plt.scatter(X_test, y_test, color='orange')
# plt.plot(X_model, y_model, color='red')
# plt.show()


def plot_learning_curves(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_errors, test_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(r2_score(y_train[:m], y_train_predict))
        test_errors.append(r2_score(y_test, y_test_predict))

    figure = plt.figure(figsize=(15, 10))
    axes = figure.add_subplot()
    _ = axes.plot(train_errors, "r-+", linewidth=1, label="train")
    _ = axes.plot(test_errors, "b-", linewidth=1, label="test")
    axes.set_ylim(ymin=0, ymax=1)
    axes.legend(fontsize=15, facecolor='#CDCDCD', labelcolor="#000000")
    axes.set_xlabel('Tamaño conjunto entrenamiento', fontsize=25, labelpad=20)
    axes.set_ylabel('R2', fontsize=25, labelpad=20)
    plt.show()

plot_learning_curves(model, X, y)