import numpy as np
from orchestrator import ExperimentOrchestrator
from utils.read_data import read_dataset
from simulation import Simulation

if __name__ == "__main__":
    X, y = read_dataset("S2")

    # orch = ExperimentOrchestrator("config.yaml", X, y)
    # orch.run_all(plot_summary=True)

    config_dict = {
        "n_client": 10,
        "k_client": 15,
        "k_server": 15,
        "partition": "dirichlet",
        "aggregation": "weighted_kmeans",
        "verbose": False,
        "r": 4,
        "snr_db": 20,
        "eps": 0,
        "sheme": "dense",
        "solver": "ols",
        "alpha": 0.2
    }
    sim = Simulation(X, y, config_dict)
    print(sim.run_trial(11))
