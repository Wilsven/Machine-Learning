import matplotlib.pyplot as plt
from sklearn import datasets

from pca import PCA

# data = datasets.load_digits()
data = datasets.load_iris()
X, y = data.data, data.target

# Project the data onto the 2 primary principal components
pca = PCA(n_components=2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X: ', X.shape)
print('Shape of transformed X: ', X_projected.shape)

x1 = X_projected[:,0] # (150, 2)
x2 = X_projected[:,1] # (150, 2)

plt.scatter(x1, x2, label=y, edgecolor='none', alpha=0.8)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

