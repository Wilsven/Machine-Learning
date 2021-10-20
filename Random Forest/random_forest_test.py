import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from random_forest import RandomForest

def accuracy(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

forest = RandomForest(n_trees=3, max_depth=10)

forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
acc = accuracy(y_test, y_pred)

print('Accuracy: ', acc)