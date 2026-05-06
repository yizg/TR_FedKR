import numpy as np
from orchestrator import ExperimentOrchestrator
from utils.read_data import read_dataset
from simulation import Simulation
from sklearn.decomposition import PCA


S_datasets = ["S1", "S2", "S3", "S4"]
pca_dim = {
    "letter": 6,
    "statlog": 3,
    "wine": 3,
    "yeast": 5}

if __name__ == "__main__":

    # Running benchmarks
    # for dataset in S_datasets:
    dataset = "S4"
    X, y = read_dataset(dataset)
    if dataset in pca_dim:
        pca = PCA(n_components=pca_dim[dataset])
        X = pca.fit_transform(X)
        print(sum(pca.explained_variance_ratio_))

    orch = ExperimentOrchestrator(
        "config_perf_k.yaml", X, y, filename=f"perf_k_{dataset}.pkl")
    orch.run_all(plot_summary=True)

    # config_dict = {
    #     "n_client": 10,
    #     "k_client": 15,
    #     "k_server": 15,
    #     "partition": "random",
    #     "aggregation": "kmeans",
    #     "weighting": "count",
    #     "verbose": False,
    #     "r": 4,
    #     "snr_db": 30,
    #     "snr_distribution": "uniform",
    #     "eps": 0,
    #     "scheme": "dense",
    #     "solver": "ols",
    #     "alpha": 0.2
    # }
    # sim = Simulation(X, y, config_dict)
    # print(sim.run_trial(11))
