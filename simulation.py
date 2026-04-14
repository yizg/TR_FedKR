import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from collections import defaultdict
from utils.math_utils import ols, random_data_partition
import utils.postprocess as postprocess
from algorithms.methods import kmeans, kmedian, trimmed_kmeans, mean_shift_filtering


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
                run_summary = postprocess.evaluation_summary(
                    self.X, server_centers, self.y)
                full_summary[run][agg] = run_summary
                self.server_centers[agg].append(server_centers)

            centralized_centers = kmeans(self.X, n_clusters=self.k_server)
            centralized_summary = postprocess.evaluation_summary(
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
