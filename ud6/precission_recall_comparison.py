import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.evaluate import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict

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
model.fit(X_train, y_train)
y_scores = cross_val_predict(model, X_train, y_train, cv = 3, method = "decision_function")
print("y_scores:")
print(y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    figure=plt.figure(figsize=(10, 5))
    axes = figure.add_subplot()
    axes.plot(thresholds, precisions[:-1], "b--", label = "Precisión")
    axes.plot(thresholds, recalls[:-1], "g-", label = "Sensibilidad")
    axes.legend(fontsize=15,facecolor='#CDCDCD',labelcolor="#000000")


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
print("Precisions:")
print(precisions)
print("Recalls:")
print(recalls)
print(thresholds)

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()