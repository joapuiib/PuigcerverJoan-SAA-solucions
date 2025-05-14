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

# Classificador binari per detectar si un dígit és un 5 o no
y_train = (y_train.astype(int) == 5)
y_test = (y_test.astype(int) == 5)

model = SGDClassifier(random_state = 42)
scores = cross_validate(model, X_train, y_train, cv=3, scoring=["accuracy", "recall", "precision"])
print(scores)
print("==============")

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

model.fit(X_train, y_train)
y_predict_test = model.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
print("Matriu de confusió:")
print(cm)
print("True negative (TN):", cm[0, 0])
print("False negative (FN):", cm[1, 0])
print("True positive (TD):", cm[1, 1])
print("False positive (FP):", cm[0, 1])
print("==============")

accuracy = accuracy_score(y_test, y_predict_test)
print(f"Accuracy: {accuracy}")
precision = precision_score(y_test, y_predict_test)
print(f"Precision: {precision}")
recall = recall_score(y_test, y_predict_test)
print(f"Recall: {recall}")
f1 = f1_score(y_test, y_predict_test)
print(f"F1: {f1}")