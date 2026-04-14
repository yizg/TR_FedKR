import numpy as np
from orchestrator import ExperimentOrchestrator
from utils.read_data import read_s_dataset


if __name__ == "__main__":
    X, y = read_s_dataset(2)
    orch = ExperimentOrchestrator("config.yaml", X, y)
    df = orch.run_all()
