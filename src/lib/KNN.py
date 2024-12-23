import numpy as np
import pandas as pd
import pickle
from scipy.stats import mode
from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm

class knn(BaseEstimator, ClassifierMixin):
    """KNN classifier with full sklearn compatibility"""
    
    def __init__(self, k=5, metric='manhattan', weights='distance', 
                 leaf_size=30, batch_size=1000, verbose=True):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.leaf_size = leaf_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.tree = None
        self.classes_ = None
        self._estimator_type = "classifier"
        
    def get_params(self, deep=True):
        """Get parameters (required for sklearn compatibility)"""
        return {
            'k': self.k,
            'metric': self.metric,
            'weights': self.weights,
            'leaf_size': self.leaf_size,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }
        
    def set_params(self, **parameters):
        """Set parameters (required for sklearn compatibility)"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _clone(self):
        """Clone estimator (required for sklearn compatibility)"""
        return knn(**self.get_params())

    def __sklearn_clone__(self):
        """Clone interface for sklearn"""
        return self._clone()
        
    def fit(self, X_train, y_train):
        """Fit KD-tree to training data"""
        # Convert input to numpy array
        self.X_train = (X_train.values if isinstance(X_train, pd.DataFrame) 
                       else X_train).astype(np.float32)
        self.y_train = np.array(y_train)
        
        # Store unique classes
        self.classes_ = np.unique(y_train)
        
        # Build KD-tree
        if self.verbose:
            print("Building KD-tree...")
        self.tree = KDTree(
            self.X_train,
            leaf_size=self.leaf_size,
            metric=self.metric
        )
        return self

    def predict(self, X_test):
        """Predict labels for test data in batches with progress bar"""
        # Convert input
        X_test = (X_test.values if isinstance(X_test, pd.DataFrame) 
                 else X_test).astype(np.float32)
        
        # Initialize predictions array
        n_samples = X_test.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        # Process in batches with progress bar
        with tqdm(total=n_samples, disable=not self.verbose) as pbar:
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                batch = X_test[i:end_idx]
                
                # Find k nearest neighbors
                distances, indices = self.tree.query(
                    batch, 
                    k=self.k,
                    return_distance=True
                )
                
                # Get corresponding labels
                neighbor_labels = self.y_train[indices]
                
                if self.weights == 'uniform':
                    # Use mode for uniform weights
                    predictions[i:end_idx] = mode(neighbor_labels, axis=1)[0].flatten()
                else:
                    # Distance-based weights
                    weights = 1 / (distances + 1e-8)  
                    weights /= np.sum(weights, axis=1, keepdims=True)
                    
                    # Weighted voting
                    for j in range(end_idx - i):
                        class_votes = np.bincount(
                            neighbor_labels[j],
                            weights=weights[j],
                            minlength=len(self.classes_)
                        )
                        predictions[i+j] = self.classes_[np.argmax(class_votes)]
                
                pbar.update(end_idx - i)
        
        return predictions

    def predict_proba(self, X_test):
        """Predict probability estimates"""
        # Convert input
        X_test = (X_test.values if isinstance(X_test, pd.DataFrame) 
                 else X_test).astype(np.float32)
        
        # Initialize probabilities array
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))
        
        # Process in batches with progress bar
        with tqdm(total=n_samples, disable=not self.verbose) as pbar:
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                batch = X_test[i:end_idx]
                
                distances, indices = self.tree.query(batch, k=self.k)
                neighbor_labels = self.y_train[indices]
                
                if self.weights == 'uniform':
                    weights = np.ones_like(distances) / self.k
                else:
                    weights = 1 / (distances + 1e-8)
                    weights /= np.sum(weights, axis=1, keepdims=True)
                
                # Calculate probabilities
                for j in range(end_idx - i):
                    for c_idx, c in enumerate(self.classes_):
                        mask = neighbor_labels[j] == c
                        probabilities[i+j, c_idx] = np.sum(weights[j][mask])
                
                pbar.update(end_idx - i)
        
        return probabilities

    def score(self, X, y):
        """Return accuracy score"""
        return np.mean(self.predict(X) == y)

    def save(self, path):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load model from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)