import numpy as np
import pandas as pd
from utils import *
from utils.math_utils import huber_regression, ols, random_data_partition, dirichlet_data_partition, random_semi_orthogonal, least_trimmed_square, theil_sen
from utils.postprocess import evaluation_summary
from algorithms.methods import kmeans, kmedian, trimmed_kmeans, mean_shift_filtering

# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────


def add_awgn(vectors, snr_db, eps=0):
    signal_power = np.mean(vectors ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    sigma1 = np.sqrt(signal_power / snr_linear)
    if eps > 0:
        sigma2 = 100 * sigma1
        impulse_mask = np.random.rand(*vectors.shape) < eps
        noise = np.where(
            impulse_mask,
            np.random.normal(0, sigma2, vectors.shape),
            np.random.normal(0, sigma1, vectors.shape)
        ).astype(np.float32)
    else:
        noise = np.random.normal(0.0, sigma1, vectors.shape).astype(np.float32)
    return vectors + noise


def denoising_process(X, r, snr_db, eps=0, scheme="rep", solver='ols', alpha=0.2):
    X_flat = X.flatten()
    d = X_flat.shape[0]

    if scheme == "rep":
        I = np.eye(d, dtype=np.float32)
        A_design = np.vstack([I] * r)
    elif scheme == "dense":
        A_design = random_semi_orthogonal(d*r, d)

    y = A_design @ X_flat
    y_noisy = add_awgn(y, snr_db, eps)

    if solver == None:
        sigma2_est = None
        X_est = y_noisy.reshape(-1, X.shape[1])
        return X_est, sigma2_est
    elif solver == 'ols':
        X_est, sigma2_est = ols(A_design, y_noisy)
    elif solver == 'lts':
        X_est, sigma2_est = least_trimmed_square(A_design, y_noisy, alpha)
    elif solver == 'huber':
        X_est, sigma2_est = huber_regression(A_design, y_noisy)
    elif solver == 'ts':
        X_est, sigma2_est = theil_sen(A_design, y_noisy, alpha)

    return X_est.reshape(X.shape), sigma2_est


# ──────────────────────────────────────────────────────────────
# Simulation Class (One config instance)
# ──────────────────────────────────────────────────────────────

class Simulation:
    def __init__(self, X, y, config: dict):
        self.X = X
        self.y = y
        # Global Settings
        self.n_client = config.get("n_client", 10)
        self.k_client = config.get("k_client", 10)
        self.k_server = config.get("k_server", 10)
        self.partition = config.get("partition", "random")
        self.aggregation = config.get("aggregation", "kmeans")
        self.verbose = config.get("verbose", False)
        # Noise parameter
        self.r = config.get("r", 1)
        self.snr_db = config.get("snr_db", 20)
        self.eps = config.get("eps", 0)
        self.scheme = config.get("scheme", "rep")
        self.solver = config.get("solver", "ols")
        self.alpha = config.get("alpha", 0.2)

        self.clients_centers = None
        self.clients_weights = None
        self.server_centers = None
        self.results = None

    def _partition_data(self, seed):
        if self.partition == "random":
            return random_data_partition(self.X, self.y, self.n_client, seed)
        else:
            return dirichlet_data_partition(self.X, self.y, self.n_client, self.partition, seed)

    def _compute_weights(self, counts, var_noise, var_cluster):
        if var_noise is not np.nan:
            var_noise_eff = self.X.shape[1] * var_noise / self.r
            var_custer_eff = var_cluster / counts
            weights = counts / (1 + var_noise_eff / var_custer_eff)  # ESS
            # weights = 1/(var_noise_eff + var_custer_eff)  # Inverse Variance
        else:
            weights = counts
        return weights.astype(np.float32)

    def _uplink_communication(self, vectors):
        vectors, var_noise_hat = denoising_process(
            vectors, self.r, self.snr_db, self.eps, self.scheme, self.solver, self.alpha)
        return vectors, var_noise_hat

    def _server_aggregate(self):
        if self.aggregation == "kmeans":
            return kmeans(self.clients_centers, n_clusters=self.k_server)
        if self.aggregation == "kmedian":
            return kmedian(self.clients_centers, n_clusters=self.k_server)
        if self.aggregation == "trimmed_kmeans":
            return trimmed_kmeans(self.clients_centers, n_clusters=self.k_server)
        if self.aggregation == "mean_shift":
            filtered = mean_shift_filtering(
                self.clients_centers, k=10, iterations=3)
            return kmeans(filtered, n_clusters=self.k_server)
        if self.aggregation == "weighted_kmeans":
            return kmeans(self.clients_centers, n_clusters=self.k_server, weights=self.clients_weights)
        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def run_trial(self, seed) -> dict:
        np.random.seed(seed)  # Reproducible per trial
        if self.n_client == 0:  # Centralized
            self.clients_centers = self.X
            self.clients_weights = None
        else:
            self.X_part, self.y_part = self._partition_data(seed)
            clients_centers, clients_weights = [], []
            for client in range(self.n_client):
                centers, counts, var_cluster = kmeans(
                    self.X_part[client], n_clusters=self.k_client, return_extra=True
                )
                centers, var_noise_hat = self._uplink_communication(centers)
                weights = self._compute_weights(
                    counts, var_noise_hat, var_cluster)
                clients_centers.append(centers)
                clients_weights.append(weights)

            self.clients_centers = np.vstack(clients_centers)
            self.clients_weights = np.hstack(clients_weights)

        self.server_centers = self._server_aggregate()
        self.results = evaluation_summary(self.X, self.server_centers, self.y)

        if self.verbose:
            print(
                f"Trial {seed} done: {[list(m.keys())[0] for m in self.server_centers.values()]}")

        return self.results
