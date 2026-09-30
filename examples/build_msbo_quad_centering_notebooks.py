"""Generate the three machine-facing msBO quadrupole-centering notebooks.

The notebooks intentionally share one workflow.  Only the PV configuration and
the original quadrupole scan differ between the three FS1 locations.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent


CONFIGS = {
    "before1stDipole": {
        "fetch_span": 4.0,
        "decision_csets": [
            "FS1_CSS:PSC2_D2276:I_CSET",
            "FS1_CSS:PSC2_D2351:I_CSET",
        ],
        "objective_rds": [
            "FS1_BBS:BPM_D2421:XPOS_RD",
            "FS1_BBS:BPM_D2466:XPOS_RD",
            "FS1_BMS:BPM_D2502:XPOS_RD",
        ],
        "quad_csets": [
            "FS1_CSS:PSQ_D2356:I_CSET",
            "FS1_CSS:PSQ_D2362:I_CSET",
            "FS1_CSS:PSQ_D2372:I_CSET",
            "FS1_CSS:PSQ_D2377:I_CSET",
        ],
        "decision_setup": dedent(
            """
            x_start = np.asarray([read_numeric_scalar(pv) for pv in decision_CSETs], dtype=float)
            decision_min = (x_start - 8.0 * AQ).tolist()
            decision_max = (x_start + 8.0 * AQ).tolist()
            decision_tols = [0.5] * len(decision_CSETs)
            """
        ),
        "state_setup": dedent(
            """
            quad_nominal = np.asarray([read_numeric_scalar(pv) for pv in state_CSETs], dtype=float)
            states = ["nominal", "low_85pct", "high_115pct"]
            state_key_vals = {
                "nominal": quad_nominal.tolist(),
                "low_85pct": (0.85 * quad_nominal).tolist(),
                "high_115pct": (1.15 * quad_nominal).tolist(),
            }
            """
        ),
    },
    "after1stDipole": {
        "fetch_span": 4.0,
        "decision_csets": [
            "FS1_CSS:PSC2_D2367:I_CSET",
            "FS1_CSS:PSC2_D2381:I_CSET",
        ],
        "objective_rds": [
            "FS1_BBS:BPM_D2466:XPOS_RD",
            "FS1_BMS:BPM_D2502:XPOS_RD",
            "FS1_BMS:BPM_D2537:XPOS_RD",
        ],
        "quad_csets": [
            "FS1_BBS:PSQ_D2416:I_CSET",
            "FS1_BBS:PSQ_D2424:I_CSET",
        ],
        "decision_setup": dedent(
            """
            x_start = np.asarray([read_numeric_scalar(pv) for pv in decision_CSETs], dtype=float)
            decision_min = (x_start - 8.0 * AQ).tolist()
            decision_max = (x_start + 8.0 * AQ).tolist()
            decision_tols = [0.5] * len(decision_CSETs)
            """
        ),
        "state_setup": dedent(
            """
            quad_nominal = np.asarray([read_numeric_scalar(pv) for pv in state_CSETs], dtype=float)
            states = ["nominal", "low_85pct", "high_115pct"]
            state_key_vals = {
                "nominal": quad_nominal.tolist(),
                "low_85pct": (0.85 * quad_nominal).tolist(),
                "high_115pct": (1.15 * quad_nominal).tolist(),
            }
            """
        ),
    },
    "after3rdDipole": {
        "fetch_span": 5.0,
        "decision_csets": [
            "FS1_BBS:PSTC_D2435:I_CSET",
            "FS1_BBS:PSTC_D2453:I_CSET",
        ],
        "objective_rds": [
            "FS1_BMS:BPM_D2502:XPOS_RD",
            "FS1_BMS:BPM_D2537:XPOS_RD",
        ],
        "quad_csets": [
            "FS1_BBS:PSQ_D2463:I_CSET",
            "FS1_BBS:PSQ_D2472:I_CSET",
        ],
        "decision_setup": dedent(
            """
            x_start = np.asarray([read_numeric_scalar(pv) for pv in decision_CSETs], dtype=float)
            ibend = read_numeric_scalar("FS1_BBS:PSD_D2394:I_CSET")
            corrector_span = abs(ibend) * 0.001 * 8.0 * AQ
            decision_min = [-corrector_span] * len(decision_CSETs)
            decision_max = [ corrector_span] * len(decision_CSETs)
            decision_tols = [0.05] * len(decision_CSETs)
            """
        ),
        "state_setup": dedent(
            """
            quad_nominal = np.asarray([read_numeric_scalar(pv) for pv in state_CSETs], dtype=float)
            states = ["nominal", "plus_5AQ"]
            state_key_vals = {
                "nominal": quad_nominal.tolist(),
                "plus_5AQ": (quad_nominal + 5.0 * AQ).tolist(),
            }
            """
        ),
    },
}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def notebook(location: str, cfg: dict) -> dict:
    filename_tag = f"[msBO][FS1][{location}]QuadCentering"
    decision_csets = repr(cfg["decision_csets"])
    objective_rds = repr(cfg["objective_rds"])
    quad_csets = repr(cfg["quad_csets"])
    config_source = "\n".join(
        [
            f"decision_CSETs = {decision_csets}",
            'decision_RDs = [pv.replace(\":I_CSET\", \":I_RD\") for pv in decision_CSETs]',
            f"objective_RDs = {objective_rds}",
            f"state_CSETs = {quad_csets}",
            'state_RDs = [pv.replace(\":I_CSET\", \":I_RD\") for pv in state_CSETs]',
            "",
            cfg["decision_setup"].strip(),
            "",
            cfg["state_setup"].strip(),
            "",
            "state_tols = [1.0] * len(state_CSETs)",
            "bpm_norms = [1.0] * len(objective_RDs)   # 1 mm scale for each BPM X reading",
            "bpm_weights = [1.0] * len(objective_RDs)",
            "",
            "config_table = pd.DataFrame({",
            '    "control": decision_CSETs,',
            '    "start": x_start,',
            '    "lower": decision_min,',
            '    "upper": decision_max,',
            '    "tolerance": decision_tols,',
            "})",
            "display(config_table)",
            "display(pd.DataFrame(state_key_vals, index=state_CSETs))",
        ]
    )

    cells = [
        markdown(
            f"""
            # FS1 quadrupole centering with multi-state Bayesian optimization

            **Location:** `{location}`

            This is the msBO replacement for the corresponding LSQ notebook.  The steering
            controls are optimized so that downstream BPM X positions change as little as
            possible when the selected quadrupoles are scanned together.  The original LSQ
            notebook is left unchanged.

            > **Machine-facing notebook:** configuration cells only read PVs, but the cells
            > marked **MOVES THE MACHINE** ramp real controls.  Review the printed bounds and
            > categorical state configurations with the operator before setting
            > `RUN_OPTIMIZATION = True`.

            Live-machine operational checklist:
            [`FS1_MACHINE_RUNBOOK.md`](FS1_MACHINE_RUNBOOK.md).
            """
        ),
        code(
            f"""
            import sys
            import json
            import datetime
            import warnings
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import torch

            # Prefer the adjacent source checkouts when running from this repository.
            # If they are absent, normal installed-package imports are attempted below.
            search_roots = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
            repo_root = next((
                root for root in search_roots
                if (root / "msBO" / "__init__.py").is_file()
                and (root / "machineIO" / "machineIO" / "__init__.py").is_file()
            ), None)
            if repo_root is not None:
                sys.path.insert(0, str(repo_root))
                sys.path.insert(0, str(repo_root / "machineIO"))

            try:
                from msBO import MultiStateBO
                from msBO.objective import QuadrupoleCentering
                from machineIO import construct_machineIO, StatefulOracleEvaluator
            except ImportError as exc:
                raise RuntimeError(
                    "Could not import msBO and machineIO. Start Jupyter from the msBO "
                    "repository or install both packages in this kernel."
                ) from exc

            machine = construct_machineIO(
                ensure_set_timeout=15,
                ensure_set_timewait_after_ramp=1.0,
                ensure_set_exception_waittime=30.0,
                fetch_data_time_span={cfg['fetch_span']},
            )

            caget = machine.caget
            seed = 0
            np.random.seed(seed)
            torch.manual_seed(seed)
            print("FS1 live-machine interface initialized.")
            print(
                "ensure_set attempts: 2 | final-timeout wait:",
                f"{{machine.ensure_set_exception_waittime:g}} s | then warn and continue",
            )
            """
        ),
        markdown(
            """
            ## Beam identity and machine configuration

            The control bounds and quadrupole scan amplitudes retain the conventions of the
            LSQ notebook.  `x_start` is captured before optimization and can be restored later.
            """
        ),
        code(
            """
            def as_epics_text(value):
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value
                if isinstance(value, (bytes, bytearray)):
                    return bytes(value).decode(errors="replace").rstrip("\\x00")
                array = np.asarray(value)
                if array.ndim and array.dtype.kind in "iu":
                    return bytes(array.astype(np.uint8).tolist()).decode(errors="replace").rstrip("\\x00")
                if array.size == 1:
                    return str(array.item())
                return str(value)

            def read_numeric_scalar(pv):
                value = caget(pv)
                if value is None:
                    raise RuntimeError(f"Required PV is unreachable: {pv}")
                try:
                    array = np.asarray(value, dtype=float).reshape(-1)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"Required PV is not numeric: {pv} -> {value!r}") from exc
                if array.size != 1 or not np.isfinite(array[0]):
                    raise RuntimeError(f"Required PV is not a finite scalar: {pv} -> {value!r}")
                return float(array[0])

            SCS = int(round(read_numeric_scalar("ACS_DIAG:DEST:ACTIVE_ION_SOURCE")))
            element = as_epics_text(caget(f"FE_ISRC{SCS}:BEAM:ELMT_BOOK"))
            if not element:
                raise RuntimeError(f"FE_ISRC{SCS}:BEAM:ELMT_BOOK returned an empty element")
            Q = int(round(read_numeric_scalar("ACC_OPS:BEAM:Q_STRIP")))
            A = int(round(read_numeric_scalar(f"FE_ISRC{SCS}:BEAM:A_BOOK")))
            if Q == 0:
                raise RuntimeError("ACC_OPS:BEAM:Q_STRIP returned zero; cannot compute A/Q")
            AQ = A / Q
            ion = f"{A}{element}{Q}"
            print(ion, "A/Q =", AQ)
            """
        ),
        code(config_source),
        markdown(
            """
            ## Read-only machine preflight

            This cell does not change any setpoint. It verifies that every required CSET,
            readback, and BPM is reachable and numeric; that decision bounds are valid; and
            that the categorical scan configurations are distinguishable at the configured
            state tolerances. It also compares the requested decision bounds with EPICS drive
            limits when those fields are available.
            """
        ),
        code(
            """
            required_numeric_pvs = list(dict.fromkeys(
                decision_CSETs + decision_RDs + state_CSETs + state_RDs + objective_RDs
            ))
            pv_snapshot = {pv: read_numeric_scalar(pv) for pv in required_numeric_pvs}

            if len(set(decision_CSETs + state_CSETs)) != len(decision_CSETs) + len(state_CSETs):
                raise RuntimeError("Decision and categorical-state CSET lists overlap or contain duplicates")
            if not np.all(np.isfinite(decision_min)) or not np.all(np.isfinite(decision_max)):
                raise RuntimeError("Decision bounds must be finite")
            if not np.all(np.asarray(decision_min) < np.asarray(decision_max)):
                raise RuntimeError("Every decision lower bound must be strictly below its upper bound")
            if not np.all((x_start >= decision_min) & (x_start <= decision_max)):
                raise RuntimeError("Captured starting controls lie outside the requested optimization bounds")

            for i, state_a in enumerate(states):
                va = np.asarray(state_key_vals[state_a], dtype=float)
                if va.shape != (len(state_CSETs),) or not np.all(np.isfinite(va)):
                    raise RuntimeError(f"Invalid categorical-state configuration: {state_a} -> {va}")
                for state_b in states[i + 1:]:
                    vb = np.asarray(state_key_vals[state_b], dtype=float)
                    if np.all(np.abs(va - vb) <= 2.0 * np.asarray(state_tols)):
                        raise RuntimeError(
                            f"State configurations {state_a!r} and {state_b!r} are not "
                            "distinguishable at twice the configured state tolerance"
                        )

            drive_limit_rows = []
            for pv, lower, upper in zip(decision_CSETs, decision_min, decision_max):
                try:
                    drvl = read_numeric_scalar(pv + ".DRVL")
                    drvh = read_numeric_scalar(pv + ".DRVH")
                except RuntimeError as exc:
                    warnings.warn(str(exc) + "; drive-limit comparison skipped")
                    continue
                drive_limit_rows.append({"PV": pv, "DRVL": drvl, "DRVH": drvh})
                if drvl < drvh and (lower < drvl or upper > drvh):
                    raise RuntimeError(
                        f"Requested bounds [{lower}, {upper}] exceed drive limits "
                        f"[{drvl}, {drvh}] for {pv}"
                    )

            display(pd.DataFrame({
                "PV": required_numeric_pvs,
                "current": [pv_snapshot[pv] for pv in required_numeric_pvs],
            }))
            if drive_limit_rows:
                display(pd.DataFrame(drive_limit_rows))
            print("Read-only preflight passed for", len(required_numeric_pvs), "required PVs")
            """
        ),
        markdown(
            """
            ## Why this BO budget?

            There are only two decision variables.  Six initial settings per state,
            `2(d+1)`, give the GP enough spatial coverage to estimate two length scales and
            task correlations without spending most of the run on initialization.  Each BO
            batch contains two candidates—the control dimension—so qEI can choose a diverse
            pair while the model is retrained half as often as in single-candidate BO.

            Four balanced rounds, `2d`, add eight adaptive settings per state. Thus every
            state receives fourteen measurements total. Each round is a closed state cycle:
            one candidate is measured while entering a state and `q-1` while leaving it. State
            order is rotated and reversed between rounds to avoid systematic ordering bias.
            """
        ),
        code(
            """
            D = len(decision_CSETs)
            S = len(states)
            J = len(objective_RDs)

            N_INIT = 2 * (D + 1)   # 6 shared Sobol control settings per state for d=2
            BATCH_SIZE = D         # q=2 jointly selected candidates
            N_ROUNDS = 2 * D       # 4 balanced BO rounds
            ACQ_STATE_MODE = "conditional"  # switch scheduler default; keep "global" as benchmark
            EXPECTED_ORACLE_CALLS = S * (N_INIT + BATCH_SIZE * N_ROUNDS)

            assert D == 2, "Revisit the budget if the number of steering controls changes."
            assert S >= 2, "Quadrupole centering needs at least two scan states."
            assert ACQ_STATE_MODE in {"global", "conditional", "mean"}
            print(f"d={D}, states={S}, BPM tasks/state={J}")
            print(f"Budget: {N_INIT} initial + {N_ROUNDS}×{BATCH_SIZE} BO calls per state")
            print(f"Expected optimization-dataset evaluations: {EXPECTED_ORACLE_CALLS}")
            """
        ),
        markdown(
            """
            ## Conditional-state qLogEI

            The state names are categorical labels—not numbers on a continuous axis. At a
            proposed control setting, the GP distinguishes the unobserved noise-free beam
            response from the noisy averaged BPM reading. Conditional-state qLogEI samples
            only readings obtainable in the scheduled state, then uses GP cross-task
            covariance and the fitted reading noise to update the mean beam response in all
            states before evaluating the quadrupole-centering objective.

            It avoids fantasy-model refits and full inner look-ahead optimization. It is the
            default for `step_batch_with_switch()` because its information model matches the
            scheduled measurement. A matched 20-seed virtual benchmark was inconclusive on
            final quality (10 wins and 10 losses versus global qLogEI), so `"global"` remains
            the required comparison mode rather than being considered obsolete.

            ![Conditional-state qLogEI flow](conditional_state_qlogei_flow.png)

            ![Matched acquisition benchmark](benchmark/conditional_state_qlogei_benchmark.png)

            Full derivation and references: [`conditional_state_qLogEI.md`](../conditional_state_qLogEI.md).
            """
        ),
        markdown(
            """
            ## Objective and oracle

            For BPM (j), msBO predicts its position at every quadrupole state.  The objective is

            \[
            f(x)=-\\frac{1}{\\sum_j w_j}\\sum_j w_j\\,\\mathrm{{Var}}_s
            \\left[\\frac{{y_{{s,j}}(x)}}{{n_j}}\\right].
            \]

            Maximizing this objective drives the quadrupole-scan slopes toward zero.  Unlike a
            generic trajectory-centering objective, it does not incorrectly demand zero position
            at downstream BPMs.  This matches the original LSQ objective's
            `var_obj_weight_fraction=1.0` behavior.
            """
        ),
        code(
            """
            oracle_key_names = {"x": decision_CSETs, "y": objective_RDs}
            oracle = StatefulOracleEvaluator(
                machine,
                control_CSETs=decision_CSETs,
                control_RDs=decision_RDs,
                control_tols=decision_tols,
                state_CSETs=state_CSETs,
                state_RDs=state_RDs,
                state_tols=state_tols,
                state_key_vals=state_key_vals,
                oracle_key_names=oracle_key_names,
                monitor_PVs=objective_RDs,
            )

            composite_objective = QuadrupoleCentering(
                S=S,
                J=J,
                norms=bpm_norms,
                weights=bpm_weights,
            )
            """
        ),
        markdown(
            """
            ## Operator gate

            The next cell is the first one that moves the machine.  It first restores the captured
            starting controls and nominal quadrupoles, then collects the initial design and runs
            balanced qEI batches. `ACQ_STATE_MODE="conditional"` matches the default switch
            scheduler behavior. It samples the incoming state's possible noisy BPM readings and
            propagates their information through GP cross-state covariance without fitting
            fantasy models. The matched 20-seed benchmark was not statistically decisive, so
            retain `"global"` as a comparison in offline validation.
            During a switch, computation prefetches only the incoming state's candidates; it never
            commits a candidate for the state after that.

            Execution has two independent operator gates: `RUN_OPTIMIZATION` and
            `LIVE_MACHINE_CONFIRMED`. Keep both false until the read-only preflight, state
            configurations, control bounds, evaluation budget, and current machine conditions
            have been reviewed.
            """
        ),
        code(
            """
            # Edit both flags only after operator review. There is no environment-variable
            # bypass because this notebook is intended exclusively for the real machine.
            RUN_OPTIMIZATION = False
            LIVE_MACHINE_CONFIRMED = False
            """
        ),
        code(
            """
            # MOVES THE MACHINE
            if not RUN_OPTIMIZATION:
                raise RuntimeError("Operator gate is closed: set RUN_OPTIMIZATION = True after review.")
            if not LIVE_MACHINE_CONFIRMED:
                raise RuntimeError(
                    "Live-machine confirmation is closed: set LIVE_MACHINE_CONFIRMED = True "
                    "only after reviewing the preflight tables and run budget."
                )

            try:
                # Establish a known nominal starting condition before MultiStateBO reads x0.
                oracle(x=x_start, s=states[0])

                msbo = MultiStateBO(
                    states=states,
                    tasks=objective_RDs,
                    control_min=decision_min,
                    control_max=decision_max,
                    control_names=decision_CSETs,
                    multistate_oracle_evaluator=oracle,
                    composite_objective_function=composite_objective,
                    local_bound_size=0.25 * (np.asarray(decision_max) - np.asarray(decision_min)),
                    asynchronous=False,
                    acq_backend="scipy",      # q-batch optimization uses BoTorch/SciPy
                    acq_restarts=8,
                    acq_raw_samples=128,
                    acq_maxiter=100,
                    fixed_mc_samples=512,
                    model_train_epochs=200,
                    model_warmstart_epochs=50,
                )

                msbo.init(n_init=N_INIT, local_optimization=False, seed=seed)

                for round_index in range(N_ROUNDS):
                    shift = round_index % S
                    state_order = states[shift:] + states[:shift]
                    if round_index % 2:
                        state_order = state_order[::-1]
                    print(f"Round {round_index + 1}/{N_ROUNDS}: {state_order}")
                    route = state_order + [state_order[0]]
                    for i in range(S):
                        msbo.step_batch_with_switch(
                            s=route[i],
                            next_s=route[i + 1],
                            prefetch_next=i < S - 1,
                            q=BATCH_SIZE,
                            local_optimization=False,
                            acq_type="EI",
                            acq_state_mode=ACQ_STATE_MODE,
                        )

                # The last switch result is ingested after its overlapped training pass.
                msbo.train_model()
            except BaseException:
                print("Optimization failed or was interrupted; attempting safe rollback.")
                try:
                    oracle(x=x_start, s=states[0])
                    print("Rollback restored starting controls and nominal quadrupoles.")
                except BaseException as rollback_error:
                    print("ROLLBACK FAILED; operator action is required:", rollback_error)
                raise

            actual_calls = len(msbo.dataset._x)
            assert actual_calls == EXPECTED_ORACLE_CALLS, (actual_calls, EXPECTED_ORACLE_CALLS)
            print("Optimization complete; oracle calls:", actual_calls)

            ensure_set_events = [
                event for event in machine.history
                if event.get("caller") == "ensure_set"
            ]
            ensure_set_retry_count = sum(
                len(event.get("attempt_statuses", [])) > 1
                for event in ensure_set_events
            )
            ensure_set_continued_timeout_count = sum(
                bool(event.get("continued_after_timeout", False))
                for event in ensure_set_events
            )
            print(
                "ensure_set retries:", ensure_set_retry_count,
                "| continued after two timeouts:", ensure_set_continued_timeout_count,
            )
            if ensure_set_continued_timeout_count:
                warnings.warn(
                    f"This run continued after {ensure_set_continued_timeout_count} "
                    "unconfirmed ramp(s). Review the machine readbacks before accepting "
                    "the recommendation.",
                    RuntimeWarning,
                )
            """
        ),
        code(
            """
            overlap = pd.DataFrame(msbo.history["time_cost"].get("switch_overlap", []))
            if not overlap.empty:
                display(overlap[[
                    "from_state", "to_state", "acq_state_mode",
                    "current_compute_sec", "compute_sec", "switch_oracle_sec",
                    "current_wait_after_compute_sec", "wait_after_compute_sec",
                ]])
                ax = overlap[["current_compute_sec", "compute_sec", "switch_oracle_sec"]].plot(
                    kind="bar", figsize=(9, 3.5)
                )
                configured_window = (
                    machine._ensure_set_timewait_after_ramp + machine._fetch_data_time_span
                )
                ax.axhline(configured_window, color="black", linestyle="--", linewidth=1,
                           label=f"configured wait + averaged reading = {configured_window:.2f} s")
                ax.set_xlabel("Switch call")
                ax.set_ylabel("Seconds")
                ax.set_title("Computation overlapped with state transition and measurement")
                ax.legend()
                plt.tight_layout()
            """
        ),
        markdown(
            """
            ## Final recommendation and validation scan

            `recommend()` maximizes the posterior mean rather than an exploratory acquisition.
            The recommended controls are then measured at every quadrupole state.  The last call
            returns the quadrupoles to nominal while retaining the recommended steering controls.
            """
        ),
        code(
            """
            # MOVES THE MACHINE
            x_recommended, predicted_objective = msbo.recommend(local_optimization=False)

            validation = {}
            before_rows = []
            try:
                for state in states:
                    result = oracle(x=x_recommended, s=state)
                    validation[state] = np.asarray(result["y"], dtype=float)
                for state in states:
                    before_rows.append(
                        np.asarray(oracle(x=x_start, s=state)["y"], dtype=float)
                    )
            finally:
                # Even if validation is interrupted, leave nominal quadrupoles and
                # the optimized steering controls applied whenever possible.
                final_nominal = oracle(x=x_recommended, s=states[0])

            validation_matrix = np.vstack([validation[s] for s in states])
            before_matrix = np.vstack(before_rows)

            comparison = pd.DataFrame({
                "BPM": objective_RDs,
                "before_state_std": before_matrix.std(axis=0),
                "after_state_std": validation_matrix.std(axis=0),
            })
            display(pd.DataFrame({"control": decision_CSETs, "start": x_start, "recommended": x_recommended}))
            display(comparison)
            print("Predicted composite objective:", predicted_objective)
            """
        ),
        markdown(
            """
            The comparison cell deliberately performs an additional complete scan at `x_start`.
            This produces a like-for-like before/after measurement under the current beam, rather
            than comparing against stale data from the start of a potentially long run.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, J, figsize=(4 * J, 3), squeeze=False)
            scan_axis = np.arange(S)
            for j, bpm in enumerate(objective_RDs):
                ax = axes[0, j]
                ax.plot(scan_axis, before_matrix[:, j], "o-", label="start controls")
                ax.plot(scan_axis, validation_matrix[:, j], "o-", label="msBO controls")
                ax.set_xticks(scan_axis, states, rotation=25, ha="right")
                ax.set_ylabel("BPM X position")
                ax.set_title(bpm.split(":", 1)[-1].replace(":XPOS_RD", ""))
                ax.grid(alpha=0.3)
                ax.legend()
            fig.suptitle("Quadrupole-scan sensitivity before and after msBO")
            fig.tight_layout()
            plt.show()

            msbo.plot_composite_objective()
            plt.show()
            """
        ),
        markdown("## Save a compact, auditable result record"),
        code(
            f"""
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            output_path = Path(f"{{now}}[{{ion}}]{filename_tag}.json")

            record = {{
                "method": "msBO qEI quadrupole centering",
                "ion": ion,
                "seed": seed,
                "acq_state_mode": ACQ_STATE_MODE,
                "states": states,
                "state_key_vals": state_key_vals,
                "decision_CSETs": decision_CSETs,
                "decision_min": decision_min,
                "decision_max": decision_max,
                "objective_RDs": objective_RDs,
                "budget": {{
                    "n_init_per_state": N_INIT,
                    "batch_size": BATCH_SIZE,
                    "rounds": N_ROUNDS,
                    "oracle_calls": len(msbo.dataset._x),
                }},
                "x_start": x_start.tolist(),
                "x_recommended": x_recommended.tolist(),
                "predicted_objective": predicted_objective,
                "validation": {{k: v.tolist() for k, v in validation.items()}},
                "before_validation": {{s: before_matrix[i].tolist() for i, s in enumerate(states)}},
                "dataset": {{
                    "x": [x.detach().cpu().tolist() for x in msbo.dataset._x],
                    "state": [states[i] for i in msbo.dataset._s],
                    "y": [y.detach().cpu().tolist() for y in msbo.dataset._y],
                }},
                "training_time_sec": msbo.history["time_cost"]["model_train"],
                "query_time_sec": msbo.history["time_cost"]["query"],
                "switch_overlap": msbo.history["time_cost"].get("switch_overlap", []),
                "machineio_ensure_set": {{
                    "retry_count": ensure_set_retry_count,
                    "continued_after_timeout_count": ensure_set_continued_timeout_count,
                    "exception_waittime_sec": machine.ensure_set_exception_waittime,
                }},
            }}
            output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print("Saved", output_path.resolve())
            """
        ),
        markdown(
            """
            ## Optional rollback

            Run only if the operator decides to discard the recommendation.  This restores both
            the captured steering controls and the nominal quadrupole state.
            """
        ),
        code(
            """
            RESTORE_START = False
            if RESTORE_START:
                oracle(x=x_start, s=states[0])
                print("Restored captured starting controls and nominal quadrupoles.")
            else:
                print("Rollback not requested; optimized controls remain applied.")
            """
        ),
    ]

    # Notebook format 4.5 requires stable cell identifiers.  Assigning them in
    # generation order keeps regenerated machine notebooks deterministic.
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for location, cfg in CONFIGS.items():
        path = HERE / f"[msBO][FS1][{location}]QuadCentering.ipynb"
        path.write_text(json.dumps(notebook(location, cfg), indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
