import numpy as np
from robpy.regression.lts import FastLTSRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor


def random_data_partition(X, y, n_client, random_state=0):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    return np.array_split(X, n_client), np.array_split(y, n_client)


def dirichlet_data_partition(X, y, n_client, alpha=1, random_state=0):
    """
    Partitions X and y into non-IID splits for clients using a Dirichlet distribution.

    Args:
        X (np.ndarray): Feature dataset.
        y (np.ndarray): Target labels.
        n_client (int): Number of clients in the federated learning setup.
        alpha (float): Dirichlet distribution parameter. Smaller = higher skew.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_splits, y_splits) where each is a list of arrays assigned to each client.
    """
    rng = np.random.default_rng(random_state)
    y = np.array(y)
    classes = np.unique(y)

    # Initialize an empty list of indices for each client
    client_indices = [[] for _ in range(n_client)]

    for c in classes:
        # 1. Get all indices in the dataset belonging to class c
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        # 2. Sample proportions for this class across all clients from a Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, n_client))

        # 3. Convert proportions into actual sample counts via cumulative sum
        cumulative_proportions = np.cumsum(proportions) * len(idx_c)
        splits = cumulative_proportions.astype(int)[:-1]

        # 4. Split the shuffled indices for class c based on the calculated splits
        idx_c_split = np.split(idx_c, splits)

        # 5. Distribute the splits to the respective clients
        for i in range(n_client):
            client_indices[i].extend(idx_c_split[i])

    # 6. Extract the actual X and y data using the grouped indices
    X_splits = []
    y_splits = []

    for i in range(n_client):
        # Shuffle the indices within each client so they aren't ordered by class
        idx = np.array(client_indices[i])
        rng.shuffle(idx)
        X_splits.append(X[idx])
        y_splits.append(y[idx])

    return X_splits, y_splits


def random_semi_orthogonal(n, k, random_state=0):
    # Initialize the generator with the seed
    rng = np.random.default_rng(random_state)

    # Use the generator instead of np.random
    Z = rng.standard_normal((n, k))

    Q, R = np.linalg.qr(Z)

    # Ensure a deterministic result by fixing the sign of the columns
    d = np.sign(np.diag(R))
    Q = Q * d
    return Q.astype(np.float32)


def ols(X, y):
    """Ordinary Least Squares regression."""
    # X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add intercept term
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    sigma2 = np.nan
    if X.shape[0]-X.shape[1] > 0:
        sigma2 = np.sum((X @ beta - y) ** 2)/(X.shape[0]-X.shape[1])
    return beta, sigma2  # [:-1] # Return coefficients and intercept


def huber_regression(X, y, epsilon=1.35):
    reg = HuberRegressor(
        epsilon=epsilon,
        alpha=0,
        fit_intercept=False,
        max_iter=1000
    ).fit(X, y)

    beta = reg.coef_

    # Robust scale estimate from HuberRegressor
    sigma = reg.scale_
    sigma2 = sigma ** 2

    return beta, sigma2


def theil_sen(X, y, alpha):
    n_sample = int((1-alpha) * X.shape[0])
    reg = TheilSenRegressor(fit_intercept=False,
                            n_subsamples=n_sample).fit(X, y)
    beta = reg.coef_
    sigma2 = np.sum((X @ beta - y) ** 2)/(X.shape[0]-X.shape[1])
    return beta, sigma2


def least_trimmed_square(X, y, alpha):
    reg = FastLTSRegression(1-alpha).fit(X, y)
    s = reg.best_h_subset
    X_filtred, y_filtred = X[s, :], y[s]
    return ols(X_filtred, y_filtred)


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
