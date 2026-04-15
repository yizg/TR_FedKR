import matplotlib.pyplot as plt
from sklearn import metrics
from utils.math_utils import assign_cluster, within_cluster_ss
import numpy as np
import pandas as pd
import plotly.express as px


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


def visualize_centers(clients_centers, server_centers, centralized_centers, weights=None):
    d = {'Client Centers': pd.DataFrame(clients_centers),
         'Server Centers': pd.DataFrame(server_centers),
         'Centralized': pd.DataFrame(centralized_centers)}
    df = pd.concat(d, axis=0).reset_index()
    if weights is not None:
        df['weights'] = pd.DataFrame(weights)
        df = df.fillna(df['weights'].mean()/2)
    fig = px.scatter(df, x=0, y=1, color='level_0', title='Server Agg Centers vs Centralized Centers', labels={
                     'level_0': 'Type'}, size='weights'if weights is not None else None)
    return fig


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


def plot_summary_bars(df):
    # self.stats['mean']['WCSS'].plot.bar(yerr=self.stats['std']['WCSS'])
    df = df.drop('WCSS', axis=1, level=0).melt(
        ignore_index=False).reset_index()
    df = df.pivot(columns=['variable_1', "experiment",],
                  index=["variable_0",], values='value')
    df = df.rename_axis(None).rename_axis([None, None], axis=1)
    df.plot(kind='bar', y='mean', yerr='std', capsize=3)
    plt.ylabel('Score')
    plt.title('Clustering Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
