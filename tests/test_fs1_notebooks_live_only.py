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
