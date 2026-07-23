"""Unit tests for the Streamlit-free shared batch step engine."""

import ast
from pathlib import Path

import pytest

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


def _timed_idle_net():
    """One timed place token that becomes ready after clock advance."""
    from cpnpy.cpn.colorsets import ColorSetParser

    cs = ColorSetParser().parse_definitions("colset TINT = int timed;")["TINT"]
    cpn = CPN()
    p1 = Place("P1", cs)
    p2 = Place("P2", cs)
    t1 = Transition("T1", variables=["x"])
    cpn.add_place(p1)
    cpn.add_place(p2)
    cpn.add_transition(t1)
    cpn.add_arc(Arc(p1, t1, "x"))
    cpn.add_arc(Arc(t1, p2, "x"))
    marking = Marking()
    marking.global_clock = 0
    # Token available at time 5
    marking.set_tokens("P1", [1], timestamps=[5])
    return cpn, marking, EvaluationContext()


def test_step_engine_module_has_no_streamlit_import():
    path = Path(__file__).resolve().parents[3] / "cpnpy" / "visualization" / "streamlit" / "step_engine.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("streamlit")
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("streamlit")


def test_macro_step_fires():
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, run_batch_macro_step

    cpn, marking, ctx = _simple_net()
    result = run_batch_macro_step(
        cpn=cpn,
        marking=marking,
        context=ctx,
        monitors=[],
        enabled_slugs=frozenset(),
        config=BatchConfig(mode="steps", max_steps=10),
        batch_firings=0,
        stop_requested=False,
    )
    assert result.kind == "fired"
    assert result.firings == 1
    assert result.info.get("transition") == "T1"
    assert len(marking.get_multiset("P1").tokens) == 2
    assert len(marking.get_multiset("P2").tokens) == 1


def test_macro_step_done_at_steps_limit():
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, run_batch_macro_step

    cpn, marking, ctx = _simple_net()
    result = run_batch_macro_step(
        cpn=cpn,
        marking=marking,
        context=ctx,
        monitors=[],
        enabled_slugs=frozenset(),
        config=BatchConfig(mode="steps", max_steps=2),
        batch_firings=2,
        stop_requested=False,
    )
    assert result.kind == "done"
    assert result.firings == 2


def test_macro_step_stopped():
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, run_batch_macro_step

    cpn, marking, ctx = _simple_net()
    result = run_batch_macro_step(
        cpn=cpn,
        marking=marking,
        context=ctx,
        monitors=[],
        enabled_slugs=frozenset(),
        config=BatchConfig(mode="steps", max_steps=10),
        batch_firings=0,
        stop_requested=True,
    )
    assert result.kind == "stopped"


def test_macro_step_advances_clock():
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, run_batch_macro_step

    cpn, marking, ctx = _timed_idle_net()
    result = run_batch_macro_step(
        cpn=cpn,
        marking=marking,
        context=ctx,
        monitors=[],
        enabled_slugs=frozenset(),
        config=BatchConfig(mode="time", target_time=10, auto_advance=True),
        batch_firings=0,
        stop_requested=False,
    )
    assert result.kind == "advanced"
    assert marking.global_clock > 0


def test_macro_step_before_monitor():
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, run_batch_macro_step

    cpn, marking, ctx = _simple_net()
    mon = MonitorSpec(
        name="Hit",
        slug=slugify_monitor_name("Hit"),
        predicate=lambda c, m: True,
        before=True,
        transition_name="T1",
        default_enabled=True,
    )
    result = run_batch_macro_step(
        cpn=cpn,
        marking=marking,
        context=ctx,
        monitors=[mon],
        enabled_slugs=frozenset({mon.slug}),
        config=BatchConfig(mode="steps", max_steps=10),
        batch_firings=0,
        stop_requested=False,
    )
    assert result.kind == "monitor"
    assert result.monitors == ["Hit"]
    assert result.info.get("transition") is None or "transition" not in result.info or not result.info.get("transition")
    # Before-monitor: token not consumed
    assert len(marking.get_multiset("P1").tokens) == 3


def test_macro_step_after_monitor():
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, run_batch_macro_step

    cpn, marking, ctx = _simple_net()
    mon = MonitorSpec(
        name="After",
        slug=slugify_monitor_name("After"),
        predicate=lambda c, m: True,
        before=False,
        transition_name="T1",
        default_enabled=True,
    )
    result = run_batch_macro_step(
        cpn=cpn,
        marking=marking,
        context=ctx,
        monitors=[mon],
        enabled_slugs=frozenset({mon.slug}),
        config=BatchConfig(mode="steps", max_steps=10),
        batch_firings=0,
        stop_requested=False,
    )
    assert result.kind == "monitor"
    assert result.monitors == ["After"]
    assert result.firings == 1
    assert len(marking.get_multiset("P2").tokens) == 1


def test_macro_step_skip_monitors_once():
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, run_batch_macro_step

    cpn, marking, ctx = _simple_net()
    mon = MonitorSpec(
        name="Hit",
        slug=slugify_monitor_name("Hit"),
        predicate=lambda c, m: True,
        before=True,
        transition_name="T1",
        default_enabled=True,
    )
    result = run_batch_macro_step(
        cpn=cpn,
        marking=marking,
        context=ctx,
        monitors=[mon],
        enabled_slugs=frozenset({mon.slug}),
        config=BatchConfig(mode="steps", max_steps=10),
        batch_firings=0,
        stop_requested=False,
        skip_monitors=True,
    )
    assert result.kind == "fired"
    assert result.firings == 1


def test_fire_transition_error_unknown():
    from cpnpy.visualization.streamlit.step_engine import fire_transition

    cpn, marking, ctx = _simple_net()
    result = fire_transition(
        cpn=cpn,
        marking=marking,
        context=ctx,
        transition_name="NoSuch",
        monitors=[],
        enabled_slugs=frozenset(),
    )
    assert result.kind == "error"
    assert result.error
