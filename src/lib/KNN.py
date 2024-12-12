import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform', verbose=True):
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
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean.")

        self.k = k
        self.verbose = verbose
        self.metric = metric
        self.weights = weights
        self.p = p if metric == 'minkowski' else (1 if metric == 'manhattan' else 2)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def _compute_nearest_neighbors(self, test_instance):
        distances = np.linalg.norm(self.X_train - test_instance, ord=self.p, axis=1)
        nearest_indices = np.argsort(distances)[:self.k]
        weights = None
        if self.weights == 'distance':
            distances = distances[nearest_indices]
            weights = 1 / (distances + 1e-5)  # Avoid division by zero
            weights /= np.sum(weights)
        return nearest_indices, weights

    def fit(self, X_train, y_train):
        self.X_train = X_train.values.astype(float) if isinstance(X_train, pd.DataFrame) else X_train.astype(float)
        self.y_train = y_train

    def _predict_single_instance(self, instance):
        nearest_indices, weights = self._compute_nearest_neighbors(instance)
        nearest_labels = self.y_train.iloc[nearest_indices] if isinstance(self.y_train, pd.Series) else self.y_train[nearest_indices]
        if self.weights == 'uniform':
            prediction = np.bincount(nearest_labels).argmax()
        else:
            prediction = np.bincount(nearest_labels, weights=weights).argmax()
        return prediction

    def predict(self, X_test):
        if self.verbose:
            print(f"Using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'} for predictions.")
        X_test = X_test.values.astype(float) if isinstance(X_test, pd.DataFrame) else X_test.astype(float)
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(executor.map(self._predict_single_instance, X_test), total=len(X_test))) if self.verbose else list(executor.map(self._predict_single_instance, X_test))
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Prediction completed in {elapsed_time:.2f} seconds.")
        return np.array(results)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)