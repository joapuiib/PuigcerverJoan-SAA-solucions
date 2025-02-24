import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
Y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7)
print(X_train.shape)
print(X_test.shape)

k_features = 5
sfs = SFS(LinearRegression(),
          k_features=k_features,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X_train, y_train)

initial_features = X_train.columns.tolist()
selected_features = list(sfs.k_feature_names_)
removed_features = list(set(initial_features)-set(selected_features))

print("Initial features: ", initial_features)
print()
print(f"## Forward selection (k = {k_features})")
print("Selected features: ", selected_features)
print("Removed features: ", removed_features)

k_features = 5
sfs = SFS(LinearRegression(),
          k_features=k_features,
          forward=False,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X_train, y_train)

selected_features = list(sfs.k_feature_names_)
removed_features = list(set(initial_features)-set(selected_features))

print()
print(f"## Backward elimination (k = {k_features})")
print("Selected features: ", selected_features)
print("Removed features: ", removed_features)

k_features = 5
sfs = SFS(LinearRegression(),
          k_features=k_features,
          forward=True,
          floating=True, # Bi-directional
          scoring = 'r2',
          cv = 0)
sfs.fit(X_train, y_train)

selected_features = list(sfs.k_feature_names_)
removed_features = list(set(initial_features)-set(selected_features))

print()
print(f"## Bidirectional forward (k = {k_features})")
print("Selected features: ", selected_features)
print("Removed features: ", removed_features)

k_features = 5
sfs = SFS(LinearRegression(),
          k_features=k_features,
          forward=False, # Backward
          floating=True, # Bi-directional
          scoring = 'r2',
          cv = 0)
sfs.fit(X_train, y_train)

selected_features = list(sfs.k_feature_names_)
removed_features = list(set(initial_features)-set(selected_features))

print()
print(f"## Bidirectional backward (k = {k_features})")
print("Selected features: ", selected_features)
print("Removed features: ", removed_features)
