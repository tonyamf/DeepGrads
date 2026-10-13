"""Helpers that load code straight out of the notebooks, so the tests exercise the real code."""
import json
import math
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def code_cells(name):
    nb = json.loads((ROOT / f"{name}.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell_containing(name, text):
    """First code cell of notebook `name` whose source contains `text`."""
    for src in code_cells(name):
        if text in src:
            return src
    raise AssertionError(f"no cell in {name} contains {text!r}")


@pytest.fixture(scope="session")
def micrograd():
    """Namespace with Value, Neuron, Layer, MLP executed from micrograd.ipynb."""
    ns = {"math": math, "random": random}
    for marker in ("class Value", "class Neuron", "class Layer", "class MLP"):
        exec(cell_containing("micrograd", marker), ns)
    return ns


@pytest.fixture(scope="session")
def words():
    return (ROOT / "names.txt").read_text().splitlines()
