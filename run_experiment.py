import yaml

with open("experiment_config.yaml") as f:
    config = yaml.safe_load(f)
print(config)

# for exp in config["experiments"]:
#     # Merge with defaults if needed
#     n_client = exp["n_client"]
#     # ... other params
#     for method in exp["aggregation_methods"]:
#         run_experiment(
#             n_client=n_client,
#             k_client=exp["k_client"],
#             # ...
#             agg_method=method,
#             run_seed=run_id  # optional
#         )
