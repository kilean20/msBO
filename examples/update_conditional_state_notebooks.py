"""Apply the CS-qLogEI documentation/API update to the two warm-start notebooks."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOKS = [
    HERE / "[VM]single-batch-warmstart.ipynb",
    HERE / "[VM]multi-batch-warmstart.ipynb",
]
MARKER = "## Conditional-state qLogEI"


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


METHOD_NOTE = r"""
## Conditional-state qLogEI

States in msBO are categorical labels; they are not continuous numeric values. At a
candidate control setting, the method samples the possible **noisy diagnostic reading**
in the scheduled state. It then uses the multi-task GP covariance and fitted reading
noise to update the mean beam response under every state before scoring the composite
objective.

This is a project-specific, same-location approximation to value-of-information—not a
full fantasy-model or knowledge-gradient calculation. The switch scheduler now defaults
to `acq_state_mode="conditional"`. This notebook also passes the mode explicitly so its
result does not depend on API defaults; that explicit selection is necessary in the
single-step notebook because `step()` retains its legacy default.

![Conditional-state qLogEI flow](conditional_state_qlogei_flow.png)

![Matched acquisition benchmark](benchmark/conditional_state_qlogei_benchmark.png)

The 20-seed virtual benchmark ended 10–10 against global qLogEI; it establishes that the
method fits the concurrent switching window, not that it universally wins. See
[`conditional_state_qLogEI.md`](../conditional_state_qLogEI.md) for the derivation,
limitations, results, and citations.
"""


TIMING_CELL = r"""
# Inspect how much switch-scheduler computation fit behind machine work.
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
    ax.axhline(4.0, color="black", linestyle="--", linewidth=1,
               label="1 s ramp + 3 s averaged reading")
    ax.set_xlabel("Switch call")
    ax.set_ylabel("Seconds")
    ax.set_title("Computation overlapped with state transition and measurement")
    ax.legend()
    plt.tight_layout()
"""


def update(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if MARKER not in source_text(cell)]

    config_index = next(
        i for i, cell in enumerate(cells)
        if cell["cell_type"] == "code" and "acq_type = 'EI'" in source_text(cell)
    )
    cells.insert(config_index, markdown(METHOD_NOTE))

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = source_text(cell)
        source = source.replace(
            "fix_acq_state = False\n",
            'acq_state_mode = "conditional"\n',
        )
        source = source.replace(
            "fix_acq_state=fix_acq_state",
            "acq_state_mode=acq_state_mode",
        )
        source = source.replace(
            "fix_acq_state = fix_acq_state",
            "acq_state_mode=acq_state_mode",
        )
        set_source(cell, source)

    if "multi-batch" in path.name:
        cells = [cell for cell in cells if "Inspect how much switch-scheduler computation" not in source_text(cell)]
        run_index = next(
            i for i, cell in enumerate(cells)
            if cell["cell_type"] == "code" and "msbo.virtual_composite_history()" in source_text(cell)
        )
        cells.insert(run_index + 1, code(TIMING_CELL))

    notebook["cells"] = cells
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    for notebook_path in NOTEBOOKS:
        update(notebook_path)
