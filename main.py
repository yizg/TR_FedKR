import numpy as np
from orchestrator import ExperimentOrchestrator
from utils.read_data import read_dataset
from simulation import Simulation

if __name__ == "__main__":
    X, y = read_dataset("S1")

    # orch = ExperimentOrchestrator("config.yaml", X, y)
    # orch.run_all(plot_summary=True)

    config_dict = {
        "n_client": 10,
        "k_client": 15,
        "k_server": 15,
        "snr_db": 20,
        "redundancy": 5,
        "aggregation": "weighted_kmeans",
        "eps": 0.01
    }
    sim = Simulation(X, y, config_dict)
    print(sim.run_trial(0))
