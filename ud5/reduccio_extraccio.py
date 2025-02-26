import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

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

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA shape:", X_train_pca.shape)
train_test_model("PCA", X_train_pca, X_test_pca, y_train, y_test)

lda = LDA(n_components=n_components)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
print(f"LDA shape:", X_train_lda.shape)
train_test_model("LDA", X_train_lda, X_test_lda, y_train, y_test)
