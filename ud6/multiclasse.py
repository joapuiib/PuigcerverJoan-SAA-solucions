import pandas as pd
from mlxtend.evaluate import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_validate

df = pd.read_csv('./../files/mnist.csv', header = None)

# Renombramos la columna 0 a target
df.rename(columns={0: 'target'}, inplace=True)

X = df.drop(columns = 'target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=91)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

n = 0
digit_n = X.loc[n, :]
print(sgd_clf.predict([digit_n]))
