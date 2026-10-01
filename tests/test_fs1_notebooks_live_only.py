import ast
from pathlib import Path

import nbformat

from examples.run_fs1_sim_smoke import instrument_for_simulation


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = [
    EXAMPLES / "[msBO][FS1][before1stDipole]QuadCentering.ipynb",
    EXAMPLES / "[msBO][FS1][after1stDipole]QuadCentering.ipynb",
    EXAMPLES / "[msBO][FS1][after3rdDipole]QuadCentering.ipynb",
]
FORBIDDEN_PRODUCTION_TERMS = (
    "simepics",
    "simulator",
    "machine_mode",
    "smoke_test",
    "msbo_run_optimization",
    "msbo_save_sim_result",
)


def notebook_source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_saved_fs1_notebooks_are_live_only_and_syntactically_valid():
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        source = notebook_source(notebook).casefold()
        assert not any(term in source for term in FORBIDDEN_PRODUCTION_TERMS)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=str(path))


def test_external_simulation_instrumentation_does_not_modify_saved_notebook():
    path = NOTEBOOKS[0]
    before = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)

    instrumented = instrument_for_simulation(notebook, smoke_test=True)
    source = notebook_source(instrumented)

    assert "start_fs1_simulator" in source
    assert "RUN_OPTIMIZATION = True" in source
    assert "LIVE_MACHINE_CONFIRMED = True" in source
    assert "N_INIT = 2" in source
    assert path.read_bytes() == before


def test_fs1_notebooks_train_on_readbacks_and_include_beam_loss_reference():
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        source = notebook_source(notebook)
        assert '"x": decision_RDs' in source
        assert '"x": decision_CSETs' not in source
        assert "BPM_MAGs_ref" in source
        assert "BPM:MAG_min_ratio" in source
        assert "beam_loss_task=True" in source
        assert "ensure_set_summary = summarize_ensure_set_history()" in source
        assert "x_recommended, predicted_objective = msbo.recommend" in source
        assert "Applied recommended controls with nominal quadrupoles." in source
        assert "regular_simplex_scan_codes" in source
        assert "len(states) == len(state_CSETs) + 1" in source
        assert "reference_states = [nominal_state, *states]" in source
        assert "scan_code_matrix.mean(axis=0)" in source
        assert "decode_individual_quad_responses" in source
        assert "s=nominal_state" in source
        assert "N_ROUNDS = D" in source
        assert "low_85pct" not in source
        assert "high_115pct" not in source
        assert "plus_5AQ" not in source
        assert "s=states[0]" not in source
        assert "rollback" not in source.casefold()
        assert "RESTORE_START" not in source
