from utils.read_data import read_s_dataset
import numpy as np
import pandas as pd
from simulation import Simulation

X, y = read_s_dataset(2)
N_CLIENT = 10
K_CLIENT = 15
K_SERVER = 15
N_RUNS = 10
PARAM_R = 2
SNR_DB = 10
AGGREGATION_METHODS = ['kmeans', 'weighted_kmeans']


print("Testing redundancy=1 with kmeans...")
# Test with redundancy=1
sim1 = Simulation(
    X=X,
    y=y,
    n_client=N_CLIENT,
    k_client=K_CLIENT,
    k_server=K_SERVER,
    n_runs=N_RUNS,
    redundancy=1,
    noise='gaussian',
    snr_db=SNR_DB,
    aggregation=AGGREGATION_METHODS
)

sim1.run()
print("\nResults for redundancy=1:")
print(sim1.stats)

print("\n" + "="*50 + "\n")

print("Testing redundancy=2 with kmeans...")
# Test with redundancy=2
sim2 = Simulation(
    X=X,
    y=y,
    n_client=N_CLIENT,
    k_client=K_CLIENT,
    k_server=K_SERVER,
    n_runs=N_RUNS,
    redundancy=2,
    noise='gaussian',
    snr_db=SNR_DB,
    aggregation=AGGREGATION_METHODS
)

sim2.run()
print("\nResults for redundancy=2:")
print(sim2.stats)

print("\n" + "="*50 + "\n")

# Compare performance
print("\nPerformance comparison (lower WCSS is better):")
print("Redundancy=1 - kmeans WCSS:", sim1.stats[('mean', 'kmeans')]['WCSS'])
print("Redundancy=1 - weighted_kmeans WCSS:",
      sim1.stats[('mean', 'weighted_kmeans')]['WCSS'])
print("Redundancy=2 - kmeans WCSS:", sim2.stats[('mean', 'kmeans')]['WCSS'])
print("Redundancy=2 - weighted_kmeans WCSS:",
      sim2.stats[('mean', 'weighted_kmeans')]['WCSS'])
