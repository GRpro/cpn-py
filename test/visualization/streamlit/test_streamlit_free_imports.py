"""Prove Streamlit-free imports for the streamlit package core."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "cpnpy" / "visualization" / "streamlit"


def _assert_no_streamlit_in_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("streamlit"), path.name
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("streamlit"), path.name


def test_core_modules_have_no_streamlit_import():
    for name in ("helpers.py", "step_engine.py", "runtime.py", "__init__.py"):
        _assert_no_streamlit_in_source(PACKAGE_ROOT / name)


def test_package_eager_import_does_not_load_streamlit():
    """Eager package import must not pull Streamlit or the UI module."""
    for key in list(sys.modules):
        if (
            key == "streamlit"
            or key.startswith("streamlit.")
            or key == "cpnpy.visualization.streamlit"
            or key == "cpnpy.visualization.streamlit.ui"
            or key.startswith("cpnpy.visualization.streamlit.cpn_graph")
        ):
            sys.modules.pop(key, None)

    mod = importlib.import_module("cpnpy.visualization.streamlit")
    assert mod.SimulationRuntime is not None
    assert mod.BatchConfig is not None
    assert "streamlit" not in sys.modules
    assert "cpnpy.visualization.streamlit.ui" not in sys.modules


def test_simulation_package_no_longer_exports_runtime():
    import cpnpy.simulation as sim

    with pytest.raises(AttributeError):
        _ = sim.SimulationRuntime
