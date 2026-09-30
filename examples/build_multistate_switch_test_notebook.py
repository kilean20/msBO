"""Generate the four-state switch-concurrency validation notebook."""

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "[VM]4state-switch-concurrency-test.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    md(
        """
        # Four-state concurrent switch/prefetch validation

        This notebook tests `step_batch_with_switch()` on a genuine multi-state route:

        ```text
        state0 → state1 → state2 → state3 → state0
        ```

        It uses a deterministic virtual oracle with configurable state-switch and measurement
        delays. The test verifies that:

        1. switch-time computation selects candidates only for the incoming state, never one state farther ahead;
        2. every state receives exactly `q` new measurements in one closed cycle;
        3. calls 2–4 consume state-compatible prefetches;
        4. GP training and acquisition computation overlap both the final current-state reading
           and the asynchronous switch measurement.

        The scheduler default is **conditional-state qLogEI**. It values only readings obtainable
        in the scheduled categorical state, then transfers their information through GP task
        covariance. An optional final cell compares it with global qLogEI and the legacy
        state-fixed mean-fill heuristic through this exact concurrent scheduler.

        ![Conditional-state qLogEI flow](conditional_state_qlogei_flow.png)
        """
    ),
    code(
        """
        import os
        import sys
        import time
        import threading
        import warnings
        from collections import Counter
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from botorch.exceptions.warnings import BadInitialCandidatesWarning

        repo_root = Path.cwd().resolve()
        if not (repo_root / "msBO" / "__init__.py").is_file():
            repo_root = repo_root.parent
        sys.path.insert(0, str(repo_root))

        from msBO import MultiStateBO
        from msBO.objective import QuadrupoleCentering
        """
    ),
    code(
        """
        class DelayedFourStateOracle:
            '''Two-control, two-output virtual machine with slow state changes.'''

            def __init__(self, switch_delay=1.5, measurement_delay=0.2, ramp_delay=0.05):
                self.states = [f"state{i}" for i in range(4)]
                self.x = np.array([0.70, 0.25], dtype=float)
                self.state = self.states[0]
                self.switch_delay = float(switch_delay)
                self.measurement_delay = float(measurement_delay)
                self.ramp_delay = float(ramp_delay)
                self.lock = threading.Lock()
                self.calls = []

            def signal(self, x, state):
                i = self.states.index(state)
                # All state-dependent terms vanish at x_center. Therefore the
                # quadrupole-centering objective has a known optimum.
                x_center = np.array([0.30, 0.65])
                dx = np.asarray(x) - x_center
                return np.array([
                    0.4 * dx[0] - 0.2 * dx[1] + (i - 1.5) * 0.8 * dx[0],
                    0.1 * dx[0] + 0.5 * dx[1] + (i - 1.5) * 0.6 * dx[1],
                ])

            def true_objective(self, x):
                y = np.stack([self.signal(x, state) for state in self.states])
                return -float(np.var(y, axis=0).mean())

            def __call__(self, x=None, s=None):
                started = time.monotonic()
                if x is None:
                    time.sleep(self.measurement_delay)
                    return {"x": self.x.copy(), "state": self.state,
                            "y": self.signal(self.x, self.state)}

                x = np.asarray(x, dtype=float)
                s = self.state if s is None else s
                if s not in self.states:
                    raise ValueError(s)

                previous_state = self.state
                time.sleep(self.ramp_delay)
                if s != previous_state:
                    time.sleep(self.switch_delay)
                time.sleep(self.measurement_delay)

                with self.lock:
                    self.x = x.copy()
                    self.state = s
                    self.calls.append({
                        "state": s,
                        "previous_state": previous_state,
                        "switched": s != previous_state,
                        "elapsed_sec": time.monotonic() - started,
                        "thread": threading.current_thread().name,
                    })
                return {"x": x, "state": s, "y": self.signal(x, s)}
        """
    ),
    code(
        """
        seed = 7
        np.random.seed(seed)
        torch.manual_seed(seed)

        oracle = DelayedFourStateOracle()
        states = oracle.states
        tasks = ["BPM0:X", "BPM1:X"]
        q = 3

        msbo = MultiStateBO(
            states=states,
            tasks=tasks,
            control_min=[0.0, 0.0],
            control_max=[1.0, 1.0],
            multistate_oracle_evaluator=oracle,
            composite_objective_function=QuadrupoleCentering(S=4, J=2),
            asynchronous=False,  # switch method has its own explicit worker overlap
            acq_backend="scipy",
            acq_restarts=4,
            acq_raw_samples=32,
            acq_maxiter=40,
            fixed_mc_samples=128,
            model_train_epochs=60,
            model_warmstart_epochs=20,
        )

        msbo.init(n_init=4, local_optimization=False, seed=seed)
        counts_before = Counter(msbo.dataset._s)
        print("Initial observations by state:", counts_before)
        """
    ),
    code(
        """
        # One closed four-state cycle. prefetch_next says whether another call
        # follows; call i prefetches q-1 points for route[i+1], never a point
        # for route[i+2].
        route = states + [states[0]]
        for i in range(len(states)):
            msbo.step_batch_with_switch(
                s=route[i],
                next_s=route[i + 1],
                prefetch_next=i < len(states) - 1,
                q=q,
                local_optimization=False,
                acq_type="EI",  # acq_state_mode omitted: exercise conditional default
            )

        # The last switch result was ingested after the overlapped training pass.
        msbo.train_model()
        """
    ),
    code(
        """
        counts_after = Counter(msbo.dataset._s)
        added = {state: counts_after[i] - counts_before[i] for i, state in enumerate(states)}
        assert added == {state: q for state in states}, added

        overlap = pd.DataFrame(msbo.history["time_cost"]["switch_overlap"])
        assert set(overlap["acq_state_mode"]) == {"conditional"}
        assert overlap["prefetch_used"].tolist() == [False, True, True, True]
        assert overlap["from_state"].tolist() == states
        assert overlap["to_state"].tolist() == states[1:] + states[:1]
        overlap["estimated_overlap_sec"] = np.minimum(
            overlap["compute_sec"], overlap["switch_oracle_sec"]
        )

        print("New measurements by state:", added)
        display(overlap[[
            "from_state", "to_state", "prefetch_next", "prefetch_used",
            "current_compute_sec", "current_wait_after_compute_sec",
            "compute_sec", "switch_oracle_sec", "wait_after_compute_sec",
            "estimated_overlap_sec",
        ]])
        """
    ),
    code(
        """
        x_recommended, predicted = msbo.recommend(local_optimization=False)
        print("Recommended x:", x_recommended)
        print("Known center:  ", np.array([0.30, 0.65]))
        print("Predicted objective:", predicted)
        print("True objective:", oracle.true_objective(x_recommended))

        fig, axes, _ = msbo.plot_composite_objective()
        fig.suptitle("Four-state concurrent msBO history")
        plt.show()
        """
    ),
    md(
        """
        ## Optional: global, conditional-state, and legacy mean-fill EI

        This is disabled by default because it runs both strategies repeatedly. It is a focused
        regression comparison for the repaired concurrent scheduler, not a replacement for the
        larger 8D/3-state benchmark suite.
        """
    ),
    code(
        """
        # Set MSBO_RUN_COMPARISON=1 when executing the notebook to run this
        # repeatable (but slower) switch-scheduler benchmark.
        RUN_GLOBAL_VS_FIXED_COMPARISON = os.getenv("MSBO_RUN_COMPARISON", "0") == "1"
        N_COMPARISON_SEEDS = int(os.getenv("MSBO_COMPARISON_SEEDS", "5"))

        def run_strategy(seed, mode):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", BadInitialCandidatesWarning)
                np.random.seed(seed)
                torch.manual_seed(seed)
                test_oracle = DelayedFourStateOracle(
                    switch_delay=0.0, measurement_delay=0.0, ramp_delay=0.0
                )
                bo = MultiStateBO(
                    states=test_oracle.states,
                    tasks=tasks,
                    control_min=[0.0, 0.0],
                    control_max=[1.0, 1.0],
                    multistate_oracle_evaluator=test_oracle,
                    composite_objective_function=QuadrupoleCentering(S=4, J=2),
                    acq_restarts=4,
                    acq_raw_samples=32,
                    acq_maxiter=40,
                    fixed_mc_samples=128,
                    model_train_epochs=60,
                    model_warmstart_epochs=20,
                )
                bo.init(n_init=4, local_optimization=False, seed=seed)
                route = test_oracle.states + [test_oracle.states[0]]
                for i in range(len(test_oracle.states)):
                    bo.step_batch_with_switch(
                        route[i], route[i + 1], q=3,
                        prefetch_next=i < len(test_oracle.states) - 1,
                        local_optimization=False,
                        acq_type="EI",
                        acq_state_mode=mode,
                    )
                bo.train_model()
                x_rec, _ = bo.recommend(local_optimization=False)
                objective = test_oracle.true_objective(x_rec)
            n_bad_init = sum(
                issubclass(item.category, BadInitialCandidatesWarning) for item in caught
            )
            return objective, n_bad_init

        if RUN_GLOBAL_VS_FIXED_COMPARISON:
            rows = []
            for comparison_seed in range(N_COMPARISON_SEEDS):
                for mode in ("global", "conditional", "mean"):
                    objective, n_bad_init = run_strategy(comparison_seed, mode)
                    rows.append({
                        "seed": comparison_seed,
                        "mode": mode,
                        "true_objective": objective,
                        "bad_initial_candidate_warnings": n_bad_init,
                    })
            comparison = pd.DataFrame(rows)
            display(comparison)
            display(comparison.groupby("mode").agg(
                mean=("true_objective", "mean"),
                median=("true_objective", "median"),
                std=("true_objective", "std"),
                bad_initial_candidate_warnings=("bad_initial_candidate_warnings", "sum"),
            ))
            paired = comparison.pivot(index="seed", columns="mode", values="true_objective")
            for mode in ("conditional", "mean"):
                paired[f"{mode}_minus_global"] = paired[mode] - paired["global"]
                print(f"{mode} pairwise wins:",
                      int((paired[f"{mode}_minus_global"] > 0).sum()), "/", len(paired))
            display(paired)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
