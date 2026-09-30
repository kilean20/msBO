"""Matched-seed benchmark for switch-scheduler acquisition semantics.

The benchmark is intentionally small enough to run as a regression test while
still exercising a four-state route through ``step_batch_with_switch``.  It
compares global qLogEI, the legacy state-fixed mean-fill approximation, and the
closed-form conditional-state qLogEI implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from scipy.stats import binomtest, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msBO import MultiStateBO
from msBO.objective import QuadrupoleCentering


class FourStateCenteringOracle:
    """Deterministic two-control quadrupole-centering surrogate."""

    def __init__(self) -> None:
        self.states = [f"state{i}" for i in range(4)]
        self.x = np.array([0.70, 0.25], dtype=float)
        self.state = self.states[0]
        self.lock = threading.Lock()

    def signal(self, x, state):
        i = self.states.index(state)
        dx = np.asarray(x, dtype=float) - np.array([0.30, 0.65])
        return np.array(
            [
                0.4 * dx[0] - 0.2 * dx[1] + (i - 1.5) * 0.8 * dx[0],
                0.1 * dx[0] + 0.5 * dx[1] + (i - 1.5) * 0.6 * dx[1],
            ]
        )

    def true_objective(self, x):
        y = np.stack([self.signal(x, state) for state in self.states])
        return -float(np.var(y, axis=0).mean())

    def __call__(self, x=None, s=None):
        if x is None:
            return {"x": self.x.copy(), "state": self.state, "y": self.signal(self.x, self.state)}
        x = np.asarray(x, dtype=float)
        s = self.state if s is None else s
        with self.lock:
            self.x = x.copy()
            self.state = s
        return {"x": x, "state": s, "y": self.signal(x, s)}


def run_one(seed: int, mode: str, args) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    oracle = FourStateCenteringOracle()
    bo = MultiStateBO(
        states=oracle.states,
        tasks=["BPM0:X", "BPM1:X"],
        control_min=[0.0, 0.0],
        control_max=[1.0, 1.0],
        multistate_oracle_evaluator=oracle,
        composite_objective_function=QuadrupoleCentering(S=4, J=2),
        acq_restarts=args.restarts,
        acq_raw_samples=args.raw_samples,
        acq_maxiter=args.maxiter,
        fixed_mc_samples=args.mc_samples,
        model_train_epochs=args.train_epochs,
        model_warmstart_epochs=args.warmstart_epochs,
        acq_seed=seed * 1000,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BadInitialCandidatesWarning)
        started = time.monotonic()
        bo.init(n_init=args.n_init, local_optimization=False, seed=seed)
        route = oracle.states + [oracle.states[0]]
        for i in range(len(oracle.states)):
            bo.step_batch_with_switch(
                route[i],
                route[i + 1],
                q=args.q,
                prefetch_next=i < len(oracle.states) - 1,
                local_optimization=False,
                acq_type="EI",
                acq_state_mode=mode,
            )
        bo.train_model()
        x_rec, _ = bo.recommend(local_optimization=False)
        elapsed = time.monotonic() - started

    overlap = bo.history["time_cost"].get("switch_overlap", [])
    query_times = np.asarray(bo.history["time_cost"].get("query", []), dtype=float)
    train_times = np.asarray(bo.history["time_cost"].get("model_train", []), dtype=float)
    return {
        "seed": seed,
        "mode": mode,
        "true_objective": oracle.true_objective(x_rec),
        "distance_to_center": float(np.linalg.norm(x_rec - np.array([0.30, 0.65]))),
        "elapsed_sec": elapsed,
        "query_total_sec": float(query_times.sum()),
        "query_median_sec": float(np.median(query_times)),
        "query_max_sec": float(query_times.max()),
        "train_total_sec": float(train_times.sum()),
        "overlap_compute_max_sec": float(max(row["compute_sec"] for row in overlap)),
        "bridge_compute_max_sec": float(max(row["current_compute_sec"] for row in overlap)),
        "bad_initial_candidate_warnings": sum(
            issubclass(item.category, BadInitialCandidatesWarning) for item in caught
        ),
    }


def summarize(frame: pd.DataFrame) -> dict:
    by_mode = frame.groupby("mode").agg(
        runs=("seed", "count"),
        objective_mean=("true_objective", "mean"),
        objective_median=("true_objective", "median"),
        objective_std=("true_objective", "std"),
        distance_median=("distance_to_center", "median"),
        elapsed_median_sec=("elapsed_sec", "median"),
        query_total_median_sec=("query_total_sec", "median"),
        query_max_median_sec=("query_max_sec", "median"),
        bridge_compute_max_median_sec=("bridge_compute_max_sec", "median"),
        overlap_compute_max_median_sec=("overlap_compute_max_sec", "median"),
        bad_initial_candidate_warnings=("bad_initial_candidate_warnings", "sum"),
    )
    paired = frame.pivot(index="seed", columns="mode", values="true_objective")
    wins = {}
    paired_statistics = {}
    if "global" in paired:
        for mode in paired.columns:
            if mode != "global":
                delta = (paired[mode] - paired["global"]).dropna()
                n_wins = int((delta > 0).sum())
                n_ties = int((delta == 0).sum())
                non_ties = delta[delta != 0]
                wins[f"{mode}_wins_vs_global"] = n_wins
                wins[f"{mode}_ties_vs_global"] = n_ties
                stats = {
                    "paired_runs": int(delta.size),
                    "median_objective_delta": float(delta.median()),
                    "mean_objective_delta": float(delta.mean()),
                    "two_sided_sign_test_p": float(
                        binomtest(n_wins, int(non_ties.size), 0.5).pvalue
                    ) if non_ties.size else 1.0,
                }
                if non_ties.size:
                    signed_rank = wilcoxon(non_ties)
                    stats["wilcoxon_statistic"] = float(signed_rank.statistic)
                    stats["wilcoxon_p"] = float(signed_rank.pvalue)
                paired_statistics[f"{mode}_vs_global"] = stats
    return {
        "by_mode": by_mode.reset_index().to_dict(orient="records"),
        "paired": wins,
        "paired_statistics": paired_statistics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=["global", "conditional", "mean"])
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--raw-samples", type=int, default=32)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--warmstart-epochs", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark" / "switch_acquisition_modes.csv",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.modes) - {"global", "conditional", "mean"})
    if unknown:
        parser.error(f"unknown modes: {unknown}")

    rows = []
    for seed in range(args.seeds):
        for mode in args.modes:
            row = run_one(seed, mode, args)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = summarize(frame)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
