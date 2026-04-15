import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from collections import defaultdict
from utils.math_utils import ols, random_data_partition
from utils.postprocess import evaluation_summary
from algorithms.methods import kmeans, kmedian, trimmed_kmeans, mean_shift_filtering

# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────


def add_awgn(vectors, snr_db=None, sigma=None):
    if snr_db is not None:
        signal_power = np.mean(vectors ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        sigma = np.sqrt(signal_power / snr_linear)
    noise = np.random.normal(0.0, sigma, vectors.shape)
    return vectors + noise


def sign_flip(vectors, flip_prob=0.01):
    mask = np.random.rand(*vectors.shape) < flip_prob
    return np.where(mask, -vectors, vectors)


def redundance_estimator(X, r, snr_db):
    X_flat = X.flatten()
    X_a = np.hstack([X_flat] * r)
    X_a_noisy = add_awgn(X_a, snr_db=snr_db)
    I = np.eye(X_flat.shape[0])
    I_a = np.vstack([I] * r)
    X_est, sigma2_est = ols(I_a, X_a_noisy)
    return X_est.reshape(X.shape), sigma2_est


# ──────────────────────────────────────────────────────────────
# Simulation Class (One config instance)
# ──────────────────────────────────────────────────────────────

class Simulation:
    def __init__(self, X, y, config: dict):
        self.X = X
        self.y = y
        self.n_client = config.get("n_client", 10)
        self.k_client = config.get("k_client", 10)
        self.k_server = config.get("k_server", 10)
        self.redundancy = config.get("param_r", 1)
        self.snr_db = config.get("snr_db", 20)
        self.flip_prob = config.get("flip_prob", 0.01)
        self.partition = config.get("partition", "random")
        self.noise = config.get("noise", "gaussian")
        self.aggregation = config.get("aggregation", "kmeans")
        self.verbose = config.get("verbose", False)

        self.clients_centers = None
        self.clients_weights = None
        self.server_centers = None
        self.results = None

    def _partition_data(self):
        if self.partition == "random":
            return random_data_partition(self.X, self.y, self.n_client)

    def _compute_weights(self, counts, var_noise=None, var_cluster=None):
        if var_noise and var_cluster:
            var_noise_eff = self.X.shape[1] * var_noise / self.redundancy
            weights = counts / (1 + var_cluster / var_noise_eff)
        else:
            weights = counts
        return weights

    def _uplink_communication(self, vectors):
        var_noise_hat = None
        if self.noise == "gaussian":
            if self.redundancy > 1:
                vectors, var_noise = redundance_estimator(
                    vectors, r=self.redundancy, snr_db=self.snr_db)
            else:
                vectors = add_awgn(vectors, snr_db=self.snr_db)
        if self.noise == "sign_flip":
            return sign_flip(vectors, flip_prob=self.flip_prob)
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
            self.X_part, self.y_part = self._partition_data()
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


# class Simulation:
#     def __init__(self, X, y,
#                  n_client=64, k_client=10, k_server=10, n_runs=10,
#                  redundancy=1,
#                  partition='random',
#                  noise='gaussian', snr_db=20, flip_prob=0.01,
#                  aggregation=['kmeans',
#                               'weighted_kmeans',
#                               'kmedian',
#                               'trimmed_kmeans',
#                               'mean_shift'],
#                  verbose=False
#                  ):
#         self.X = X
#         self.y = y
#         self.n_runs = n_runs
#         self.partition = partition
#         self.n_client = n_client
#         self.k_client = k_client
#         self.k_server = k_server
#         self.noise = noise
#         self.snr_db = snr_db
#         self.flip_prob = flip_prob
#         self.aggregation = aggregation
#         self.redundancy = redundancy

#         self.stats = None
#         self.clients_centers = []
#         self.clients_weights = []
#         self.server_centers = defaultdict(list)
#         self.centralized_centers = []

#         self.verbose = verbose

#     def partition_data(self):
#         if self.partition == 'random':
#             return random_data_partition(self.X, self.y, self.n_client)

#     def noisy_communication(self, vectors):
#         if self.noise == 'gaussian':
#             return add_awgn(vectors, snr_db=self.snr_db)
#         elif self.noise == 'sign_flip':
#             return sign_flip(vectors, flip_prob=self.flip_prob)
#         else:
#             return vectors

#     def server_aggregate(self, centers, agg_method, weights=None):
#         if agg_method == 'kmeans':
#             server_centers = kmeans(centers, n_clusters=self.k_server)
#         elif agg_method == 'kmedian':
#             server_centers = kmedian(centers, n_clusters=self.k_server)
#         elif agg_method == 'trimmed_kmeans':
#             server_centers = trimmed_kmeans(centers, n_clusters=self.k_server)
#         elif agg_method == 'mean_shift':
#             centers = mean_shift_filtering(centers, k=10, iterations=3)
#             server_centers = kmeans(centers, n_clusters=self.k_server)
#         elif agg_method == 'weighted_kmeans':
#             server_centers = kmeans(
#                 centers, n_clusters=self.k_server, weights=weights)
#         return server_centers

#     def run(self):
#         full_summary = defaultdict(dict)

#         # 1. Partition data for each client
#         X_part, y_part = self.partition_data()

#         for run in range(self.n_runs):
#             clients_centers = []
#             clients_weights = []
#             for client in range(self.n_client):

#                 # 2. Each client runs local k-means
#                 centers, counts, variances = kmeans(X_part[client],
#                                                     n_clusters=self.k_client,
#                                                     return_extra=True)

#                 # 3. Clients send their centers through a noisy channel
#                 if self.redundancy > 1:
#                     # Use redundancy estimator for redundancy > 1
#                     centers, var_noise = redundance_estimator(
#                         centers, r=self.redundancy, snr_db=self.snr_db)
#                     var_noise_eff = centers.shape[1] * \
#                         var_noise / self.redundancy
#                     # Calculate weights based on counts and noise variance
#                     weights = counts / (1 + variances / var_noise_eff)
#                 else:
#                     # For redundancy = 1, use simple noisy communication
#                     centers = self.noisy_communication(centers)
#                     # For redundancy = 1, we still need weights for weighted_kmeans
#                     # Use counts as weights (more points in cluster = more weight)
#                     weights = counts.copy()

#                 clients_centers.append(centers)
#                 clients_weights.append(weights)

#             clients_centers = np.vstack(clients_centers)
#             clients_weights = np.hstack(clients_weights)
#             self.clients_centers.append(clients_centers)
#             self.clients_weights.append(clients_weights)

#             # 4. Server aggregates client centers and compute performance statistics
#             for agg in self.aggregation:
#                 server_centers = self.server_aggregate(
#                     clients_centers, agg_method=agg, weights=clients_weights)
#                 run_summary = postprocess.evaluation_summary(
#                     self.X, server_centers, self.y)
#                 full_summary[run][agg] = run_summary
#                 self.server_centers[agg].append(server_centers)

#             centralized_centers = kmeans(self.X, n_clusters=self.k_server)
#             centralized_summary = postprocess.evaluation_summary(
#                 self.X, centralized_centers, self.y)
#             full_summary[run]['Centralized K-Means'] = centralized_summary
#             self.centralized_centers.append(centralized_centers)

#             if self.verbose:
#                 print(f"Run: {run+1}/{self.n_runs} {run_summary}")

#         combined = pd.concat({k: pd.DataFrame(v)
#                              for k, v in full_summary.items()}, axis=1).stack()
#         means = combined.mean(axis=1).unstack()
#         stds = combined.std(axis=1).unstack()
#         self.stats = pd.concat({'mean': means, 'std': stds}, axis=1)

#         self.centralized_centers = np.array(self.centralized_centers)
#         self.clients_centers = np.array(self.clients_centers)
#         return

#     def plot_summary(self):
#         if self.stats is None:
#             raise ValueError(
#                 "No summary available. Please run the simulation first.")
#         # self.stats['mean']['WCSS'].plot.bar(yerr=self.stats['std']['WCSS'])
#         self.stats.drop('WCSS').plot(
#             kind='bar', y='mean', yerr='std', capsize=3)
#         plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
#         plt.title('Clustering Performance Comparison')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()
