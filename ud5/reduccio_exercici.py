import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('./../files/ud5/laptop_pricing_cleaned.csv', index_col=0)
df = df.iloc[:, 1:]
df = df.select_dtypes(include=['number'])

X = df.drop(columns = 'Price', axis = 1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size   = 0.7)

# scaler = preprocessing.MinMaxScaler()
# X_train = scaler.fit_transform(X_train)

# print(X_train)

# Filtre univariant - Variables costants (VarianceThreshold)
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train)

print(X_train.var())
print("Columnas originales: ", len(X_train.columns))
print("Columnas després del threshold: ", len(X_train.columns[constant_filter.get_support()]))
print(X_train.columns[constant_filter.get_support()])

# Filtre multivariant - Variables correlacionades entre si
figure = plt.figure(figsize = (15, 10))
axes = figure.add_subplot()
corr_features = X_train.corr()
_ = sns.heatmap(corr_features, annot=True, cmap=plt.cm.coolwarm, axes = axes)
# plt.show()

corr_threshold = 0.7
corr_matrix = X_train.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
print(upper)

to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
print(to_drop)

# Drop features
X_train = X_train.drop(columns=to_drop, axis=1)
X_test = X_test.drop(columns=to_drop, axis=1)

print(X_train.shape)