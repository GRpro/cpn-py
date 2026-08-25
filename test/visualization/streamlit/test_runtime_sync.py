"""Sync SimulationRuntime tests (Streamlit-free)."""

import ast
import copy
from pathlib import Path

from cpnpy.cpn.cpn_imp import (
    CPN,
    Place,
    Transition,
    Arc,
    Marking,
    EvaluationContext,
)
from cpnpy.cpn.colorsets import IntegerColorSet
from cpnpy.visualization.streamlit.helpers import MonitorSpec, slugify_monitor_name


def _simple_net(tokens=(1, 2, 3)):
    cpn = CPN()
    p1 = Place("P1", IntegerColorSet())
    p2 = Place("P2", IntegerColorSet())
    t1 = Transition("T1", variables=["x"])
    cpn.add_place(p1)
    cpn.add_place(p2)
    cpn.add_transition(t1)
    cpn.add_arc(Arc(p1, t1, "x"))
    cpn.add_arc(Arc(t1, p2, "x"))
    marking = Marking()
    marking.add_tokens("P1", list(tokens))
    return cpn, marking, EvaluationContext()


def test_runtime_module_has_no_streamlit_import():
    path = Path(__file__).resolve().parents[3] / "cpnpy" / "visualization" / "streamlit" / "runtime.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("streamlit")
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("streamlit")


def test_marking_identity_stable_across_fires():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _simple_net()
    rt = SimulationRuntime(cpn, marking, ctx)
    mid = id(rt.marking)
    rt.fire("T1")
    rt.fire("T1")
    assert id(rt.marking) == mid
    assert len(rt.marking.get_multiset("P2").tokens) == 2


def test_run_batch_sync_steps():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _simple_net()
    rt = SimulationRuntime(cpn, marking, ctx)
    status = rt.run_batch_sync(BatchConfig(mode="steps", max_steps=2))
    assert status.terminal_reason == "done"
    assert status.firings == 2
    assert not status.async_running


def test_reset_marking_restores_initial():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime

    cpn, marking, ctx = _simple_net()
    rt = SimulationRuntime(cpn, marking, ctx)
    rt.fire("T1")
    assert len(rt.marking.get_multiset("P1").tokens) == 2
    rt.reset_marking()
    assert len(rt.marking.get_multiset("P1").tokens) == 3
    assert id(rt.marking) == id(marking)


def test_sync_before_monitor_and_skip_once():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _simple_net()
    mon = MonitorSpec(
        name="Hit",
        slug=slugify_monitor_name("Hit"),
        predicate=lambda c, m: True,
        before=True,
        transition_name="T1",
        default_enabled=True,
    )
    rt = SimulationRuntime(cpn, marking, ctx, monitors=[mon])
    status = rt.run_batch_sync(BatchConfig(mode="steps", max_steps=10))
    assert status.terminal_reason == "monitor"
    assert status.monitor_names == ("Hit",)
    assert len(rt.marking.get_multiset("P1").tokens) == 3

    rt.skip_monitors_once()
    status = rt.run_batch_sync(BatchConfig(mode="steps", max_steps=10))
    assert status.firings >= 1
    assert len(rt.marking.get_multiset("P2").tokens) >= 1
