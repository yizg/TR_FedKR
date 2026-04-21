from algorithms.kmedian import KMedian
from algorithms.kFED import kfed
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np


def kmeans(X, n_clusters, weights=None, return_extra=False):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X, sample_weight=weights)
    if return_extra:
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        counts = np.bincount(labels)
        variances = np.zeros(n_clusters, dtype=np.float32)
        for i in range(n_clusters):
            mask = (labels == i)
            cluster_points = X[mask]
            center = centers[i]
            squared_distances = np.sum((cluster_points - center) ** 2, axis=1)
            # Total variance across D dim
            variances[i] = np.mean(squared_distances)
        return centers, counts, variances

    return kmeans.cluster_centers_


def kmedian(X, n_clusters):
    kmedian = KMedian(n_clusters=n_clusters)
    kmedian.fit(X)
    return kmedian.cluster_centers_


def trimmed_kmeans(X, n_clusters, trim_ratio=0.1, max_iter=300, tol=1e-4, random_state=None):
    """
    Trimmed K-Means clustering algorithm.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_clusters : int
        Number of clusters to form
    trim_ratio : float, default=0.1
        Proportion of points to trim (remove) in each iteration
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int or None, default=None
        Random seed for initialization

    Returns:
    --------
    cluster_centers : array, shape (n_clusters, n_features)
        Final cluster centers
    """
    n_samples, n_features = X.shape
    n_trim = int(n_samples * trim_ratio)

    # Initialize centers using k-means++ initialization
    rng = np.random.default_rng(random_state)

    # Select first center randomly
    centers = np.zeros((n_clusters, n_features))
    centers[0] = X[rng.integers(n_samples)]

    # K-means++ initialization for remaining centers
    for i in range(1, n_clusters):
        # Compute distances to nearest center for each point
        distances = np.min(
            np.sum((X[:, np.newaxis] - centers[:i]) ** 2, axis=2), axis=1)
        # Sample next center with probability proportional to distance squared
        probs = distances / distances.sum()
        centers[i] = X[rng.choice(n_samples, p=probs)]

    for iteration in range(max_iter):
        # Compute distances from each point to each center
        distances = np.sum((X[:, np.newaxis] - centers) ** 2, axis=2)

        # Assign each point to nearest center
        labels = np.argmin(distances, axis=1)

        # Compute distances to assigned centers
        point_distances = distances[np.arange(n_samples), labels]

        # Sort points by distance and trim farthest points
        sorted_indices = np.argsort(point_distances)
        keep_indices = sorted_indices[:-
                                      n_trim] if n_trim > 0 else sorted_indices

        # Update centers using only kept points
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            cluster_points = X[(labels == k) & np.isin(
                np.arange(n_samples), keep_indices)]
            if len(cluster_points) > 0:
                new_centers[k] = cluster_points.mean(axis=0)
            else:
                # If cluster becomes empty, reinitialize with a random kept point
                new_centers[k] = X[keep_indices[rng.integers(
                    len(keep_indices))]]

        # Check for convergence
        center_shift = np.sqrt(np.sum((new_centers - centers) ** 2))
        centers = new_centers

        if center_shift < tol:
            break

    return centers


def mean_shift_filtering(X, k=10, iterations=1, include_self=False,):
    """Mean shift filtering using k-nearest neighbors."""
    X = np.asarray(X)
    n_samples = X.shape[0]

    # Ensure k is not larger than number of samples
    if k > n_samples:
        raise ValueError(
            f"k ({k}) cannot exceed number of samples ({n_samples})")
    if not include_self and k > n_samples - 1:
        raise ValueError(
            f"Without self, k cannot exceed n_samples-1 ({n_samples-1})")

    X_current = X.copy()

    for _ in range(iterations):
        # Build nearest neighbors model on the current positions
        # Use algorithm='auto' (usually kd_tree or ball_tree for low/medium dims)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nbrs.fit(X_current)

        # Find indices of k nearest neighbors for each point
        # distances, indices = nbrs.kneighbors(X_current)
        indices = nbrs.kneighbors(X_current, return_distance=False)

        # For each point, compute the mean of its neighbors
        new_points = []
        for i in range(n_samples):
            neighbor_idxs = indices[i]
            # If include_self is False, ensure we remove the point itself
            if not include_self:
                # neighbor_idxs may contain i; remove it
                neighbor_idxs = neighbor_idxs[neighbor_idxs != i]
                # If after removal we have fewer than k neighbors (shouldn't happen by check)
                if len(neighbor_idxs) == 0:
                    # Fallback: keep the point unchanged
                    new_points.append(X_current[i])
                    continue
            mean_vec = np.mean(X_current[neighbor_idxs], axis=0)
            new_points.append(mean_vec)

        X_current = np.array(new_points)

    return X_current
