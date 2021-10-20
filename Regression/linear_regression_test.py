import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from base_regression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X, y, color='b', marker='o', s=30)
# plt.show()

# Define function to calculate MSE
def mse(y_true, y_predicted):
    return  np.mean((y_true - y_predicted)**2)

# With default lr=0.001
regressor_default = LinearRegression()
regressor_default.fit(X_train, y_train)
y_predicted_default = regressor_default.predict(X_test)

print(mse(y_test, y_predicted_default))

# With lr=0.01
regressor_lr = LinearRegression(lr=0.01)
regressor_lr.fit(X_train, y_train)
y_predicted_lr = regressor_lr.predict(X_test)

print(mse(y_test, y_predicted_lr))

# # Plot predicted linear regression line
# y_pred_line = regressor.predict(X)
# fig = plt.figure(figsize=(8, 6))
# m1 = plt.scatter(X_train, y_train, color='b', s=10)
# m2 = plt.scatter(X_test, y_test, color='r', s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.show()

