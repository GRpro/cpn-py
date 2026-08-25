"""Streamlit simulation UI package — public entry for visualizer and runtime.

Eager exports are Streamlit-free (``SimulationRuntime``, ``BatchConfig``, helpers).
``CPNStreamlitVisualizer`` is available via lazy ``__getattr__`` so headless
imports never pull Streamlit.
"""

from __future__ import annotations

from typing import Any

from cpnpy.visualization.streamlit.helpers import (
    FAST_BATCH_MAX_ITERATIONS,
    MonitorSpec,
    format_simulation_error,
    raise_if_invalid_net,
    slugify_monitor_name,
    validate_net_for_simulation,
)
from cpnpy.visualization.streamlit.runtime import (
    DEFAULT_LEASE_S,
    RuntimeStatus,
    SimulationRuntime,
)
from cpnpy.visualization.streamlit.step_engine import BatchConfig, StepResult

__all__ = [
    "BatchConfig",
    "CPNStreamlitVisualizer",
    "DEFAULT_LEASE_S",
    "FAST_BATCH_MAX_ITERATIONS",
    "MonitorSpec",
    "RuntimeStatus",
    "SimulationRuntime",
    "StepResult",
    "format_simulation_error",
    "raise_if_invalid_net",
    "slugify_monitor_name",
    "validate_net_for_simulation",
]

_LAZY_ATTRS = frozenset({"CPNStreamlitVisualizer"})


def __getattr__(name: str) -> Any:
    if name == "CPNStreamlitVisualizer":
        from cpnpy.visualization.streamlit.ui import CPNStreamlitVisualizer

        return CPNStreamlitVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
