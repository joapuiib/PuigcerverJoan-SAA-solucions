import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import LinearRegression
from sklearn import metrics

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

# Keep only 1000 samples
# X = X[:1000]
# y = y[:1000]

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, train_size=0.7)

df = X_train_full.copy()
df['target'] = y_train_full

# print("X_train shape:", X_train.shape, "({0:.2f})".format((X_train.shape[0] * 100) / X.shape[0]))
# print("X_val shape:", X_val.shape, "({0:.2f})".format((X_val.shape[0] * 100) / X.shape[0]))
# print("X_test shape:", X_val.shape, "({0:.2f})".format((X_test.shape[0] * 100) / X.shape[0]))
errors = np.empty(shape=[100])

for i in range(0, 100):
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, train_size=0.7)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_val)
    MSE = metrics.mean_squared_error(y_val, y_predict)
    errors[i] = MSE

print("N-MSE: Máximo: %.2f. Mínimo: %.2f" % (-errors.max(), -errors.min()))

## Leave One Out Cross-Validation (LOOCV)
from sklearn.model_selection import LeaveOneOut
cv = LeaveOneOut()

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
# cv = KFold(n_splits=10)
cv = RepeatedKFold(n_splits=10, n_repeats=10)

from sklearn.model_selection import cross_validate
model = LinearRegression()
scoring = ['neg_mean_squared_error', 'r2']
scores = cross_validate(model, X_train_full, y_train_full,
                        cv=cv,
                        scoring=scoring)

print("%s N-MSE: mean = %.2f" % (cv.__class__.__name__ , scores['test_neg_mean_squared_error'].mean()))
print("%s R2: mean = %.2f" % (cv.__class__.__name__ , scores['test_r2'].mean()))

model.fit(X_train_full, y_train_full)

y_predict = model.predict(X_train_full)
train_MSE = metrics.mean_squared_error(y_train_full, y_predict)
train_r2 = metrics.r2_score(y_train_full, y_predict)
print("LOOCV N-MSE: train = %.2f" % (-train_MSE))
print("LOOCV R2: train = %.2f" % (train_r2))
y_predict = model.predict(X_test)
test_MSE = metrics.mean_squared_error(y_test, y_predict)
test_r2 = metrics.r2_score(y_test, y_predict)
print("LOOCV N-MSE: test = %.2f" % (-test_MSE))
print("LOOCV R2: test = %.2f" % (test_r2))
