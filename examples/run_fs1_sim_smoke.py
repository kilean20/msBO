"""Test live-only FS1 notebooks using an in-memory simEPICS instrumentation.

The saved notebooks are never modified. Each in-memory copy receives a local
IOC bootstrap, open operator gates, shortened machine timing, suppressed result
writes, and (unless ``--full-budget`` is used) a reduced BO/GP budget.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


if sys.platform == "win32":
    # pyzmq requires add_reader(), which the default Proactor loop does not
    # implement.  Selecting this policy avoids an extra compatibility thread
    # and its warning in the notebook smoke-test harness.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NOTEBOOKS = [
    HERE / "[msBO][FS1][before1stDipole]QuadCentering.ipynb",
    HERE / "[msBO][FS1][after1stDipole]QuadCentering.ipynb",
    HERE / "[msBO][FS1][after3rdDipole]QuadCentering.ipynb",
]


SIMULATOR_BOOTSTRAP = """
from examples.fs1_simulation import start_fs1_simulator
simulator = start_fs1_simulator(seed=2026)
""".strip()


SIMULATOR_SHUTDOWN = """
simulator.stop()
print("External localhost IOC fixture stopped.")
""".strip()


def instrument_for_simulation(notebook, *, smoke_test: bool):
    """Return an in-memory notebook configured for localhost validation."""
    notebook.cells.insert(1, nbformat.v4.new_code_cell(SIMULATOR_BOOTSTRAP))

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source

        # The external IOC must use PyEPICS and short acquisition timings. This
        # transformation applies only to the in-memory copy executed below.
        source = source.replace("ensure_set_timeout=15,", "ensure_set_timeout=3,")
        source = source.replace(
            "ensure_set_timewait_after_ramp=1.0,",
            "ensure_set_timewait_after_ramp=0.01,",
        )
        source = source.replace(
            "ensure_set_exception_waittime=30.0,",
            "ensure_set_exception_waittime=0.1,",
        )
        for live_span in ("4.0", "5.0"):
            source = source.replace(
                f"fetch_data_time_span={live_span},",
                "fetch_data_time_span=0.15,\n"
                "    sample_interval=0.05,\n"
                "    use_epics=True,\n"
                "    isOK_PVs=[\"ACS_DIAG:CHP:STATE_RD\"],\n"
                "    isOK_vals=[3],",
            )

        source = source.replace("RUN_OPTIMIZATION = False", "RUN_OPTIMIZATION = True")
        source = source.replace("LIVE_MACHINE_CONFIRMED = False", "LIVE_MACHINE_CONFIRMED = True")

        if smoke_test:
            source = source.replace("N_INIT = 2 * (D + 1)", "N_INIT = 2")
            source = source.replace("N_ROUNDS = 2 * D", "N_ROUNDS = 1")
            source = source.replace("acq_restarts=8,", "acq_restarts=2,")
            source = source.replace("acq_raw_samples=128,", "acq_raw_samples=16,")
            source = source.replace("acq_maxiter=100,", "acq_maxiter=20,")
            source = source.replace("fixed_mc_samples=512,", "fixed_mc_samples=32,")
            source = source.replace("model_train_epochs=200,", "model_train_epochs=20,")
            source = source.replace("model_warmstart_epochs=50,", "model_warmstart_epochs=8,")

        source = source.replace(
            'output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")\n'
            'print("Saved", output_path.resolve())',
            'print("External simulator result not written:", output_path.name)',
        )
        cell.source = source

    notebook.cells.append(nbformat.v4.new_code_cell(SIMULATOR_SHUTDOWN))
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-budget",
        action="store_true",
        help="Use the live-size BO budget while still talking only to simEPICS.",
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")

    for path in NOTEBOOKS:
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        notebook = instrument_for_simulation(
            notebook,
            smoke_test=not args.full_budget,
        )
        NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        ).execute()
        print(f"PASS {path.name} ({time.monotonic() - started:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
