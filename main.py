import numpy as np
from orchestrator import ExperimentOrchestrator
from utils.read_data import read_dataset
from simulation import Simulation

if __name__ == "__main__":
    X, y = read_dataset("statlog")

    # orch = ExperimentOrchestrator("config.yaml", X, y)
    # orch.run_all(plot_summary=True)

    config_dict = {
        "n_client": 10,
        "k_client": 6,
        "k_server": 6,
        "snr_db": 50,
        "redundancy": 1,
        "aggregation": "kmeans",
        "eps": 0,
    }
    sim = Simulation(X, y, config_dict)
    print(sim.run_trial(0))
