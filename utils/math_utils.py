import numpy as np


def random_data_partition(X, y, n_client, random_state=0):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    return np.array_split(X, n_client), np.array_split(y, n_client)


def ols(X, y):
    """Ordinary Least Squares regression."""
    # X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add intercept term
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    sigma2 = np.sum((X @ beta - y) ** 2)/(X.shape[0]-X.shape[1])
    return beta, sigma2  # [:-1] # Return coefficients and intercept


def compute_dist(X, centers):
    """Compute the squared distance between each point and each center."""
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    return sq_dist


def within_cluster_ss(X, centers):
    """Compute the within-cluster sum of squares for a given set of centers."""
    sq_dist = compute_dist(X, centers)
    return np.sum(np.min(sq_dist, axis=1))


def assign_cluster(X, centers):
    """Assign each point to the closest center."""
    sq_dist = compute_dist(X, centers)
    labels = np.argmin(sq_dist, axis=1)
    return labels
