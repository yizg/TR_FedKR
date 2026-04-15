import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from simulation import Simulation
import pickle
from datetime import datetime
from utils.postprocess import plot_summary_bars


class ExperimentOrchestrator:
    def __init__(self, config_path: str, X: np.ndarray, y: np.ndarray, plot_summary=False):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        self.defaults = raw.get("defaults", {})
        self.experiments = raw.get("experiments", [])
        self.X = X
        self.y = y
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.output_prefix = raw.get("output_prefix", "result")
        self.configs = []
        self.plot_summary = plot_summary

    def _merge_config(self, exp: dict) -> dict:
        cfg = self.defaults.copy()
        cfg.update(exp)
        return cfg

    def _build_filename(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.results_dir / f"{self.output_prefix}_{timestamp}.pkl"

    def run_all(self) -> pd.DataFrame:
        all_results = []

        for exp in self.experiments:
            exp_name = exp.get("name", "unnamed")
            cfg = self._merge_config(exp)
            self.configs.append(cfg)
            n_runs = cfg.get("n_runs", 10)
            base_seed = cfg.get("base_seed", 0)

            print(f"\n Experiment: {exp_name} | n_runs={n_runs}")
            sim = Simulation(self.X, self.y, cfg)

            for run_idx in range(n_runs):
                seed = base_seed + run_idx
                print(f"Run {run_idx+1}/{n_runs} (seed={seed})")
                try:
                    trial_res = sim.run_trial(seed)
                    row = {"experiment": exp_name,
                           "run_idx": run_idx,
                           "seed": seed,
                           **trial_res}
                    all_results.append(row)
                except Exception as e:
                    print(f"  Run {run_idx+1} failed: {e}")

        df = pd.DataFrame(all_results)
        summary = self._save_and_summarize(df)
        if self.plot_summary:
            plot_summary_bars(summary)
        return summary

    def _save_and_summarize(self, df: pd.DataFrame):
        output = {}
        filename = self._build_filename()
        output['raw_results'] = df
        output['config'] = self.configs
        # Compute mean/std per experiment
        num_cols = df.select_dtypes(include=np.number).columns
        summary = df.groupby("experiment")[num_cols].agg(
            ["mean", "std"]).reset_index().drop(["run_idx", "seed"], axis=1, level=0).set_index('experiment')
        output['summary'] = summary

        with open(filename, 'wb') as f:
            pickle.dump(output, f)

        print(summary.to_string(index=False))
        print(f"\n Summary saved to `{filename}`")
        return summary
