import numpy as np
from typing import Tuple, List


class KMedian:
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels = None
        self.inertia = None

    def fit(self, X: np.ndarray) -> 'KMedian':
        n_samples, n_features = X.shape

        # Initialize centroids using random data points
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()

        for iteration in range(self.max_iter):
            # Assign labels based on Manhattan distance
            labels = self._assign_labels(X)

            # Compute new centroids as medians of clusters
            new_centroids = np.zeros_like(self.cluster_centers_)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.median(cluster_points, axis=0)
                else:
                    new_centroids[i] = self.cluster_centers_[i]

            # Check for convergence
            if np.all(np.abs(new_centroids - self.cluster_centers_) < self.tol):
                break

            self.cluster_centers_ = new_centroids

        self.labels = labels
        self.inertia = self._compute_inertia(X)
        return self

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sum(
                np.abs(X - self.cluster_centers_[i]), axis=1)
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X: np.ndarray) -> float:
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum(np.abs(cluster_points -
                                  self.cluster_centers_[i]))
        return inertia

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_labels(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels
