import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time
from scipy.stats import mode

class knn:
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform', 
                 verbose=True, batch_size=100):  # Reduced batch size
        # Input validation
        if k < 1 or not isinstance(k, int):
            raise ValueError("k must be a positive integer.")
        if metric not in ['manhattan', 'euclidean', 'minkowski']:
            raise ValueError("Invalid metric. Choose from 'manhattan', 'euclidean', or 'minkowski'.")
        if p < 1 or not isinstance(p, (int, float)):
            raise ValueError("p must be a positive number.")
        if weights not in ['uniform', 'distance']:
            raise ValueError("weights must be either 'uniform' or 'distance'.")
        if n_jobs < 1 and n_jobs != -1 or not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be a positive integer or -1.")

        self.k = k
        self.verbose = verbose
        self.metric = metric
        self.weights = weights
        self.p = p if metric == 'minkowski' else (1 if metric == 'manhattan' else 2)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.batch_size = batch_size

    def _compute_distances(self, X1, X2):
        """
        Memory-efficient distance computation
        """
        if self.metric == 'euclidean':
            # Compute distances without creating large intermediate arrays
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                distances[i] = np.sqrt(np.sum((X2 - X1[i]) ** 2, axis=1))
            return distances
        elif self.metric == 'manhattan':
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                distances[i] = np.sum(np.abs(X2 - X1[i]), axis=1)
            return distances
        else:  # minkowski
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                distances[i] = np.power(np.sum(np.power(np.abs(X2 - X1[i]), self.p), axis=1), 1/self.p)
            return distances

    def fit(self, X_train, y_train):
        self.X_train = X_train.values.astype(np.float32) if isinstance(X_train, pd.DataFrame) else X_train.astype(np.float32)
        self.y_train = np.array(y_train)
        return self

    def _predict_batch(self, X_batch):
        # Compute distances for the batch
        distances = self._compute_distances(X_batch, self.X_train)
        
        # Get indices of k nearest neighbors
        nearest_neighbor_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        
        # Get corresponding labels
        nearest_labels = self.y_train[nearest_neighbor_indices]
        
        if self.weights == 'uniform':
            predictions = mode(nearest_labels, axis=1)[0].flatten()
        else:
            k_distances = np.take_along_axis(distances, nearest_neighbor_indices, axis=1)
            weights = 1 / (k_distances + 1e-5)
            weights /= np.sum(weights, axis=1, keepdims=True)
            
            predictions = np.zeros(X_batch.shape[0], dtype=int)
            for i in range(X_batch.shape[0]):
                predictions[i] = np.bincount(nearest_labels[i], 
                                          weights=weights[i], 
                                          minlength=len(np.unique(self.y_train))).argmax()
        
        return predictions

    def predict(self, X_test):
        if self.verbose:
            print(f"Using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'} for predictions.")
        
        X_test = X_test.values.astype(np.float32) if isinstance(X_test, pd.DataFrame) else X_test.astype(np.float32)
        
        # Process in smaller chunks sequentially to avoid memory issues
        n_samples = X_test.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in tqdm(range(0, n_samples, self.batch_size), disable=not self.verbose):
            end_idx = min(i + self.batch_size, n_samples)
            batch = X_test[i:end_idx]
            predictions[i:end_idx] = self._predict_batch(batch)
        
        return predictions

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)