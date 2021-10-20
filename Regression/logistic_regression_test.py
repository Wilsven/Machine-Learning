import numpy as np
from scipy.sparse.construct import rand 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from base_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

def accuracy(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

log_reg = LogisticRegression(lr=0.0001)
log_reg.fit(X_train, y_train)
y_predicted = log_reg.predict(X_test)

print(accuracy(y_test, y_predicted))