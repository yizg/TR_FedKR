import numpy as np
import pandas as pd
from kmedian import KMedian
from sklearn.cluster import KMeans
from utils import *
from kFED import kfed
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


def random_data_partition(X, y, n_client, random_state=0):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    return np.array_split(X, n_client), np.array_split(y, n_client)


def kmeans(X, n_clusters, weights=None, return_extra=False):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X, sample_weight=weights)
    if return_extra:
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        counts = np.bincount(labels)
        variances = np.zeros(n_clusters)
        for i in range(n_clusters):
            mask = (labels == i)
            cluster_points = X[mask]
            center = centers[i]
            squared_distances = np.sum((cluster_points - center) ** 2, axis=1)
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


def ols(X, y):
    """Ordinary Least Squares regression."""
    # X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add intercept term
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    sigma2 = np.sum((X @ beta - y) ** 2)/(X.shape[0]-X.shape[1])
    return beta, sigma2  # [:-1] # Return coefficients and intercept


def redundance_estimator(X, r=2, snr_db=20):
    X_flat = X.flatten()
    X_a = np.hstack([X_flat for _ in range(r)])
    X_a_noisy = add_awgn(X_a, snr_db=snr_db)
    I = np.eye(X_flat.shape[0])
    I_a = np.vstack([I for _ in range(r)])
    X_est, sigma2_est = ols(I_a, X_a_noisy)  # variance per dimensions
    return X_est.reshape(X.shape), sigma2_est


def add_awgn(vectors, snr_db=None, sigma=None):
    """Additive white Gaussian noise (AWGN) to a signal."""
    if snr_db is not None:
        signal_power = np.mean(vectors ** 2)
        # SNR in linear scale
        snr_linear = 10 ** (snr_db / 10.0)
        # Noise power = signal_power / snr_linear
        noise_power = signal_power / snr_linear
        sigma = np.sqrt(noise_power)
    # print(sigma, vectors.shape)
    # Generate Gaussian noise of same shape as vectors
    noise = np.random.normal(loc=0.0, scale=sigma, size=vectors.shape)
    noisy = vectors + noise
    return noisy


def sign_flip(vectors, flip_prob=0.01):
    """Randomly flip the sign of elements in the vectors with a given probability."""
    mask = np.random.rand(*vectors.shape) < flip_prob
    flipped = np.where(mask, -vectors, vectors)
    return flipped


class Simulation:
    def __init__(self, X, y,
                 n_client=64, k_client=10, k_server=10, n_runs=10,
                 redundancy=1,
                 partition='random',
                 noise='gaussian', snr_db=20, flip_prob=0.01,
                 aggregation=['kmeans',
                              'weighted_kmeans',
                              'kmedian',
                              'trimmed_kmeans',
                              'mean_shift'],
                 verbose=False
                 ):
        self.X = X
        self.y = y
        self.n_runs = n_runs
        self.partition = partition
        self.n_client = n_client
        self.k_client = k_client
        self.k_server = k_server
        self.noise = noise
        self.snr_db = snr_db
        self.flip_prob = flip_prob
        self.aggregation = aggregation
        self.redundancy = redundancy

        self.stats = None
        self.clients_centers = []
        self.clients_weights = []
        self.server_centers = defaultdict(list)
        self.centralized_centers = []

        self.verbose = verbose

    def partition_data(self):
        if self.partition == 'random':
            return random_data_partition(self.X, self.y, self.n_client)

    def noisy_communication(self, vectors):
        if self.noise == 'gaussian':
            return add_awgn(vectors, snr_db=self.snr_db)
        elif self.noise == 'sign_flip':
            return sign_flip(vectors, flip_prob=self.flip_prob)
        else:
            return vectors

    def server_aggregate(self, centers, agg_method, weights=None):
        if agg_method == 'kmeans':
            server_centers = kmeans(centers, n_clusters=self.k_server)
        elif agg_method == 'kmedian':
            server_centers = kmedian(centers, n_clusters=self.k_server)
        elif agg_method == 'trimmed_kmeans':
            server_centers = trimmed_kmeans(centers, n_clusters=self.k_server)
        elif agg_method == 'mean_shift':
            centers = mean_shift_filtering(centers, k=10, iterations=3)
            server_centers = kmeans(centers, n_clusters=self.k_server)
        elif agg_method == 'weighted_kmeans':
            server_centers = kmeans(
                centers, n_clusters=self.k_server, weights=weights)
        return server_centers

    def run(self):
        full_summary = defaultdict(dict)

        # 1. Partition data for each client
        X_part, y_part = self.partition_data()

        for run in range(self.n_runs):
            clients_centers = []
            clients_weights = []
            for client in range(self.n_client):

                # 2. Each client runs local k-means
                centers, counts, variances = kmeans(X_part[client],
                                                    n_clusters=self.k_client,
                                                    return_extra=True)

                # 3. Clients send their centers through a noisy channel
                if self.redundancy > 1:
                    # Use redundancy estimator for redundancy > 1
                    centers, var_noise = redundance_estimator(
                        centers, r=self.redundancy, snr_db=self.snr_db)
                    var_noise_eff = centers.shape[1] * \
                        var_noise / self.redundancy
                    # Calculate weights based on counts and noise variance
                    weights = counts / (1 + variances / var_noise_eff)
                else:
                    # For redundancy = 1, use simple noisy communication
                    centers = self.noisy_communication(centers)
                    # For redundancy = 1, we still need weights for weighted_kmeans
                    # Use counts as weights (more points in cluster = more weight)
                    weights = counts.copy()

                clients_centers.append(centers)
                clients_weights.append(weights)

            clients_centers = np.vstack(clients_centers)
            clients_weights = np.hstack(clients_weights)
            self.clients_centers.append(clients_centers)
            self.clients_weights.append(clients_weights)

            # 4. Server aggregates client centers and compute performance statistics
            for agg in self.aggregation:
                server_centers = self.server_aggregate(
                    clients_centers, agg_method=agg, weights=clients_weights)
                run_summary = evaluation_summary(
                    self.X, server_centers, self.y)
                full_summary[run][agg] = run_summary
                self.server_centers[agg].append(server_centers)

            centralized_centers = kmeans(self.X, n_clusters=self.k_server)
            centralized_summary = evaluation_summary(
                self.X, centralized_centers, self.y)
            full_summary[run]['Centralized K-Means'] = centralized_summary
            self.centralized_centers.append(centralized_centers)

            if self.verbose:
                print(f"Run: {run+1}/{self.n_runs} {run_summary}")

        combined = pd.concat({k: pd.DataFrame(v)
                             for k, v in full_summary.items()}, axis=1).stack()
        means = combined.mean(axis=1).unstack()
        stds = combined.std(axis=1).unstack()
        self.stats = pd.concat({'mean': means, 'std': stds}, axis=1)

        self.centralized_centers = np.array(self.centralized_centers)
        self.clients_centers = np.array(self.clients_centers)
        return

    def plot_summary(self):
        if self.stats is None:
            raise ValueError(
                "No summary available. Please run the simulation first.")
        # self.stats['mean']['WCSS'].plot.bar(yerr=self.stats['std']['WCSS'])
        self.stats.drop('WCSS').plot(
            kind='bar', y='mean', yerr='std', capsize=3)
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Clustering Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
