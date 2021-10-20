import numpy as np


class LDA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None
        
    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        # Within class scatter matrix:
        # S_W = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # S_B = sum( n_c * (mean_X_c - mean_overall)^2 )
        
        mean_overall = np.mean(X, axis=0) # (1, 4)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0) # (1, 4)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            S_W += (X_c - mean_c).T.dot((X_c - mean_c)) # (4, 4)
            
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1) # (4, 1)
            # (4, 1) * (4, 1).T = 4, 1) * (1, 4) = (4,4) -> reshape
            S_B += n_c * mean_diff.dot(mean_diff.T) # (4, 4)
            
        # Determine SW^-1 * SB
        A = np.linalg.inv(S_W.dot(S_B))
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)  
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low  
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        
        eigenvalues, eigenvectors = eigenvalues[idxs], eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[:self.n_components]
              
    def transform(self, X):
        # Project data 
        return np.dot(X, self.linear_discriminants.T)