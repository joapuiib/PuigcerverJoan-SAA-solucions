import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X = np.array([-6, 5, 10 ,2, 0])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X.reshape(-1, 1))
print(X_norm)

mu = np.mean(X)
sigma = np.std(X)

X_std = (X - mu) / sigma
print("Media:", mu)
print("Desviación típica:", sigma)
print(X)
print(X_std)