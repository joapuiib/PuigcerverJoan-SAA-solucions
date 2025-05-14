import numpy as np

from sklearn.svm import SVC

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = "bwr"

X, y = make_moons(n_samples=100, noise=0.15)

svc_model = SVC(kernel="linear")
svc_model.fit(X, y)


def draw_model(X, y, model, axes):
    axes.set_xlabel('X1', size='x-large')
    axes.set_ylabel('X2', size='x-large')

    _ = axes.scatter(X[:, 0], X[:, 1], c=y)

    h = .02
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    _ = axes.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

# hiper_params = np.array([
#     [4, 1],
#     [4, 100],
#     [10, 1],
#     [10, 100]
# ])
#
# figure = plt.figure(figsize=(20, 15), constrained_layout=True)
# for index, param in enumerate(hiper_params):
#     svc_model = SVC(kernel="poly", degree=param[0], coef0=param[1], C=5)
#     svc_model.fit(X, y)
#     axes = figure.add_subplot(2, 2, index + 1)
#     title = "d = " + str(param[0]) + ", r = " + str(param[1]) + ", C = 5"
#     axes.set_title(title, fontsize=15, pad=20, color="#003B80")
#     draw_model(X, y, svc_model, axes)

hiper_params = np.array([
    [1, 1],
    [1, 1000],
    [5, 1],
    [5, 1000]
])

figure = plt.figure(figsize=(20, 15), constrained_layout=True)
for index, param in enumerate(hiper_params):
    svc_model = SVC(kernel="rbf", gamma=param[0], C=param[1])
    svc_model.fit(X, y)
    axes = figure.add_subplot(2, 2, index + 1)
    title = "gamma = " + str(param[0]) + ", C = " + str(param[1])
    axes.set_title(title, fontsize=15, pad=20, color="#003B80")
    draw_model(X, y, svc_model, axes)
plt.show()
