"""
Execute every analysis notebook top-to-bottom and assert no cell raises.

These tests are SLOW (minutes each) and require the real .mat data files
in ../data_raw/.  They are excluded from the default test run.

Run only notebook tests:
    cd analyses && python -m pytest tests/test_notebooks.py -v -m notebook

Run everything *except* notebook tests:
    cd analyses && python -m pytest tests/ -v -m "not notebook"

Run all tests including notebooks:
    cd analyses && python -m pytest tests/ -v
"""

import os
import pytest
import nbformat
from nbclient import NotebookClient

ANALYSES_DIR = os.path.join(os.path.dirname(__file__), "..")

# Every .ipynb in the analyses directory
NOTEBOOKS = [
    "individual_diff.ipynb",
    "cross_task.ipynb",
    "cross_task_3way.ipynb",
    "cross_epoch.ipynb",
    "individual_3Dgeometries.ipynb",
    "figure.ipynb",              # depends on data from all tasks; run last
]

# Timeout per cell in seconds (some cells do 100-iteration CV loops)
CELL_TIMEOUT = 600


def _run_notebook(name):
    """Execute a notebook and return it.  Raises on any cell error."""
    path = os.path.join(ANALYSES_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not found")

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": ANALYSES_DIR}},
    )
    client.execute()
    return nb


@pytest.mark.notebook
class TestNotebooks:
    """Each method executes one notebook top-to-bottom."""

    def test_individual_diff(self):
        _run_notebook("individual_diff.ipynb")

    def test_cross_task(self):
        _run_notebook("cross_task.ipynb")

    def test_cross_task_3way(self):
        _run_notebook("cross_task_3way.ipynb")

    def test_cross_epoch(self):
        _run_notebook("cross_epoch.ipynb")

    def test_individual_3d_geometries(self):
        _run_notebook("individual_3Dgeometries.ipynb")

    def test_figure(self):
        _run_notebook("figure.ipynb")
