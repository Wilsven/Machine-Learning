import os
import sys

PROJECTPATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'Decision Tree')
sys.path.append(PROJECTPATH)

import numpy as np
from collections import Counter
from decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):
    most_common_label = Counter(y).most_common(1)[0][0]
    return most_common_label


class RandomForest:
    
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
         
    def fit(self, X, y):
        self.tress = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, 
                                max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_predictions = [most_common_label(tree_prediction) for tree_prediction in tree_predictions]
        return np.array([y_predictions])
        
        





