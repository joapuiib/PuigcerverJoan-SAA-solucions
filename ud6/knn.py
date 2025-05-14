import pandas as pd
import numpy as np
from scipy import stats

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


def nearest_neighbors(a, X_train, y_train, k):
    distances = X_train.apply(lambda x: euclidean_distance(a, x), axis = 1).sort_values()
    indexes = distances[0:k].index
    results = y_train[indexes]
    result = stats.mode(results, axis=None)
    return result.mode


iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

X = df.drop(columns=['target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
y_predict_manual = X_test.apply(lambda x: nearest_neighbors(x, X_train, y_train, k), axis=1)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_predict_sklearn = model.predict(X_test)

print("Confusion matrix for manual implementation")
print(confusion_matrix(y_test, y_predict_manual))

print("Confusion matrix for sklearn implementation")
print(confusion_matrix(y_test, y_predict_sklearn))