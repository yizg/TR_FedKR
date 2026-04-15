import numpy as np
from orchestrator import ExperimentOrchestrator
from utils.read_data import read_s_dataset, read_letter_dataset


if __name__ == "__main__":
    X, y = read_letter_dataset()
    orch = ExperimentOrchestrator("config.yaml", X, y)
    orch.run_all(plot_summary=True)
