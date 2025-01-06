import numpy as np
from collections import defaultdict
import pickle
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin

class gnb(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes with sklearn compatibility"""
    def __init__(self, batch_size=1000, verbose=True):
        self.batch_size = batch_size
        self.verbose = verbose
        self._estimator_type = "classifier"
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None
        self.n_features_ = None
        self.fitted_ = False
    
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
        """Fit Gaussian Naive Bayes classifier"""
        # Convert input
        X = X.values if hasattr(X, 'values') else np.array(X)
        y = np.array(y)
        
        # Store classes first
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        
        # Compute class priors
        class_counts = np.bincount(y)
        self.class_priors_ = class_counts / len(y)
        
        # Initialize parameters
        n_classes = len(self.classes_)
        self.means_ = np.zeros((n_classes, self.n_features_))
        self.variances_ = np.zeros((n_classes, self.n_features_))
        
        # Compute means and variances for each class
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.variances_[i] = X_c.var(axis=0) + 1e-9
        
        self.fitted_ = True
        return self

    def _compute_log_likelihood_batch(self, X_batch):
        """Compute log likelihood for a batch of samples"""
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        log_likelihood = np.zeros((X_batch.shape[0], len(self.classes_)))
        
        for i, _ in enumerate(self.classes_):
            diff = X_batch - self.means_[i]
            exponent = -0.5 * np.sum(diff ** 2 / self.variances_[i], axis=1)
            log_likelihood[:, i] = np.log(self.class_priors_[i]) + exponent
                
        return log_likelihood

    def predict(self, X):
        """Predict class labels"""
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
            
        X = X.values if hasattr(X, 'values') else np.array(X)
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=self.classes_.dtype)
        
        with tqdm(total=n_samples, disable=not self.verbose) as pbar:
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X[i:end_idx]
                
                log_likelihood = self._compute_log_likelihood_batch(X_batch)
                predictions[i:end_idx] = self.classes_[np.argmax(log_likelihood, axis=1)]
                
                pbar.update(end_idx - i)
        
        return predictions

    def predict_proba(self, X):
        """Predict probability estimates"""
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X.values if hasattr(X, 'values') else np.array(X)
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, len(self.classes_)))
        
        with tqdm(total=n_samples, disable=not self.verbose) as pbar:
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X[i:end_idx]
                
                log_likelihood = self._compute_log_likelihood_batch(X_batch)
                probas[i:end_idx] = np.exp(log_likelihood - np.max(log_likelihood, axis=1, keepdims=True))
                probas[i:end_idx] /= probas[i:end_idx].sum(axis=1, keepdims=True)
                
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