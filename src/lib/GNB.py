import numpy as np
from collections import defaultdict
import pickle
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin

class gnb(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes with sklearn compatibility"""
    def __init__(self, batch_size=1000, verbose=True):
        # Initialize parameters
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Protected attributes
        self._estimator_type = "classifier"
        self._const = None
        self._log_const = None
        
        # Model state
        self.classes_ = None  # Changed from classes to classes_
        self.class_priors_ = None  # Changed from class_priors
        self.means_ = None  # Changed from means
        self.variances_ = None  # Changed from variances
        self.n_features_ = None  # Changed from n_features
    
    def get_params(self, deep=True):
        """Get parameters (required for sklearn compatibility)"""
        return {
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
        return gnb(**self.get_params())

    def __sklearn_clone__(self):
        """Clone interface for sklearn"""
        return self._clone()

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes classifier
        Vectorized implementation for faster training
        """
        # Convert input to numpy array if needed
        X = X.values if hasattr(X, 'values') else np.array(X)
        y = np.array(y)
        
        # Store dimensions
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        
        # Compute class priors
        class_counts = np.bincount(y)
        self.class_priors = class_counts / len(y)
        
        # Initialize parameters
        self.means = np.zeros((len(self.classes), self.n_features))
        self.variances = np.zeros((len(self.classes), self.n_features))
        
        # Compute means and variances for each class
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i] = X_c.mean(axis=0)
            self.variances[i] = X_c.var(axis=0) + 1e-9  # Add small constant for numerical stability
        
        # Pre-compute constants for Gaussian density
        self._const = 1 / np.sqrt(2 * np.pi * self.variances)
        self._log_const = np.log(self._const)
        
        return self

    def _compute_log_likelihood_batch(self, X_batch):
        """
        Compute log likelihood for a batch of samples
        Vectorized implementation for faster prediction
        """
        # Initialize log likelihood matrix
        log_likelihood = np.zeros((X_batch.shape[0], len(self.classes)))
        
        # Add log priors
        log_likelihood += np.log(self.class_priors)
        
        # Compute log likelihood for each class
        for i, _ in enumerate(self.classes):
            # Vectorized computation of Gaussian density
            diff = X_batch - self.means[i]
            exponent = -0.5 * np.sum(diff ** 2 / self.variances[i], axis=1)
            log_likelihood[:, i] += np.sum(self._log_const[i]) + exponent
            
        return log_likelihood

    def predict(self, X):
        """
        Predict class labels for samples in X
        Uses batch processing and parallel computation for large datasets
        """
        # Convert input to numpy array if needed
        X = X.values if hasattr(X, 'values') else np.array(X)
        
        # Process data in batches
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in tqdm(range(n_batches)):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            
            # Compute log likelihoods for batch
            log_likelihood = self._compute_log_likelihood_batch(X_batch)
            
            # Get predictions for batch
            predictions[start_idx:end_idx] = self.classes[np.argmax(log_likelihood, axis=1)]
        
        return predictions
    
    def predict_proba(self, X):
        """Predict probability estimates"""
        X = X.values if hasattr(X, 'values') else np.array(X)
        
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, len(self.classes_)))
        
        with tqdm(total=n_samples, disable=not self.verbose) as pbar:
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X[i:end_idx]
                
                log_likelihood = self._compute_log_likelihood_batch(X_batch)
                probas[i:end_idx] = np.exp(log_likelihood - np.max(log_likelihood, axis=1, keepdims=True))
                probas[i:end_idx] /= np.sum(probas[i:end_idx], axis=1, keepdims=True)
                
                pbar.update(end_idx - i)
                
        return probas

    def score(self, X, y):
        """Return accuracy score"""
        return np.mean(self.predict(X) == y)

    def save(self, path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)