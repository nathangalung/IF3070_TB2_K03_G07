import numpy as np
from collections import defaultdict
import pickle


class GNB:
    def __init__(self):
        self.class_priors = None
        self.means = None
        self.variances = None

    def fit(self, X, y):
        self.class_priors = self._compute_class_priors(y)
        self.means, self.variances = self._compute_means_and_variances(X, y)

    def _compute_class_priors(self, y):
        class_counts = defaultdict(int)
        total_samples = len(y)

        for label in y:
            class_counts[label] += 1

        class_priors = {
            label: count / total_samples for label, count in class_counts.items()
        }
        return class_priors

    def _compute_means_and_variances(self, X, y):
        unique_classes = np.unique(y)
        means = {}
        variances = {}

        for label in unique_classes:
            class_data = X[y == label]
            means[label] = np.mean(class_data, axis=0)
            variances[label] = np.var(class_data, axis=0)

        return means, variances

    def _gaussian_density(self, x, mean, variance):
        epsilon = 1e-10
        variance = max(variance, epsilon)
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponent

    def _compute_log_likelihood(self, features, label):
        log_likelihood = np.log(self.class_priors[label])

        for i, feature in enumerate(features):
            mean = self.means[label][i]
            variance = self.variances[label][i]
            log_likelihood += np.log(self._gaussian_density(feature, mean, variance))

        return log_likelihood

    def predict(self, X):
        predictions = []

        for sample in X.values:
            log_likelihoods = {
                label: self._compute_log_likelihood(sample, label)
                for label in self.class_priors
            }

            predicted_class = max(log_likelihoods, key=log_likelihoods.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)