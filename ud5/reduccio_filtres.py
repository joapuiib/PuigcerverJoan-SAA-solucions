import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from ud5.reduccio_exercici import corr_matrix

wine = datasets.load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Original shape:", X_train.shape)

def train_test_model(name, X_train, X_test, y_train, y_test):
    model_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    model_knn.fit(X_train, y_train)

    y_pred_rf = model_knn.predict(X_test)
    print(name, model_knn.__class__.__name__, float("{0:.4f}".format(accuracy_score(y_test, y_pred_rf))))

n_components = 2

# Variance Threshold
scaler = MinMaxScaler()
X_train_vt = scaler.fit_transform(X_train)
X_test_vt = scaler.transform(X_test)

X_train_vt_df = pd.DataFrame(X_train_vt, columns=X_train.columns)
variances = X_train_vt_df.var()
variances = variances.sort_values(ascending=False)

filter = VarianceThreshold(threshold=0.065)
X_train_vt = filter.fit_transform(X_train_vt)
X_test_vt = filter.transform(X_test_vt)

print("VARIANCE THRESHOLD")
print(f"VarianceThreshold shape:", X_train_vt.shape, X_train.columns)
train_test_model("VarianceThreshold", X_train_vt, X_test_vt, y_train, y_test)

# Correlation
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Get the name of the column and index of that cell
min_corr = upper.stack().idxmin()

# Keep min_corr[0] and min_corr[1] columns
X_train_corr = X_train[[min_corr[0], min_corr[1]]]
X_test_corr = X_test[[min_corr[0], min_corr[1]]]

print("CORRELATION")
print(f"Correlation shape:", X_train_corr.shape, min_corr)
train_test_model("Correlation", X_train_corr, X_test_corr, y_train, y_test)


# PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA shape:", X_train_pca.shape)
train_test_model("PCA", X_train_pca, X_test_pca, y_train, y_test)

# LDA
lda = LDA(n_components=n_components)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
print(f"LDA shape:", X_train_lda.shape)
train_test_model("LDA", X_train_lda, X_test_lda, y_train, y_test)
