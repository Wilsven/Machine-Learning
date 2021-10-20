import numpy as np


class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Calculate mean, mean centering
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        
        # Calculate covariance, function needs samples as columns
        cov = np.cov(X.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvectors)[::-1] # reverse 
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # Store first n eigenvectors
        self.components = eigenvectors[:self.n_components]
         
    def transform(self, X):
        # Project data 
        X -= self.mean
        return np.dot(X, self.components.T)