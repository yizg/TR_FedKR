import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


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


def plot_assign_cluster_2d(data, centers):
    labels = assign_cluster(data, centers)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='red',
                marker='x', s=200, label="Centroid")
    plt.title('Assigned clusters')
    plt.legend()


def plot_clusters_2d(data, labels, centers=None):
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red',
                    marker='x', s=200, label='Centroid')
    plt.legend()


def evaluation_summary(X, centers, true_labels=None):
    """
    Evaluate the clustering results using various metrics.

    Parameters:
    - X: The original data points (n_samples, n_features).
    - centers: Final cluster centers found by the server (n_clusters, n_features).
    - true_labels: The true labels for the data points (optional).
    """
    output = {}
    n, d = X.shape
    labels = assign_cluster(X, centers)

    wcss = within_cluster_ss(X, centers)
    output['WCSS'] = wcss
    # nmse = wcss / (n * d)
    # output['nMSE'] = nmse

    if true_labels is not None:
        ari = metrics.adjusted_rand_score(true_labels, labels)
        output['ARI'] = ari
        nmi = metrics.normalized_mutual_info_score(true_labels, labels)
        output['NMI'] = nmi
        confusion_matrix = metrics.confusion_matrix(true_labels, labels)
        purity = np.sum(np.amax(confusion_matrix, axis=0)) / \
            np.sum(confusion_matrix)
        output['Purity'] = purity

    silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    output['Silhouette'] = silhouette
    # calinski_harabasz = metrics.calinski_harabasz_score(X, labels)
    # output['C-H Ratio'] = calinski_harabasz
    return output
