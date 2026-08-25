"""Async SimulationRuntime batch tests (Streamlit-free)."""

import threading
import time

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


def _chain_net(n_tokens=50):
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
    marking.add_tokens("P1", list(range(n_tokens)))
    return cpn, marking, EvaluationContext()


def _wait_terminal(rt, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = rt.poll_status(renew=False)
        if not st.async_running:
            return st
        time.sleep(0.01)
    raise AssertionError("async batch did not terminate in time")


def test_async_large_steps_without_streamlit():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _chain_net(40)
    rt = SimulationRuntime(cpn, marking, ctx)
    assert rt.submit_non_animated_batch(BatchConfig(mode="steps", max_steps=40))
    status = _wait_terminal(rt)
    assert status.terminal_reason == "done"
    assert status.firings == 40
    assert len(rt.marking.get_multiset("P2").tokens) == 40


def test_async_cooperative_stop():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _chain_net(200)

    def slow_guard_net():
        # Slow each fire slightly so stop has a window
        original = cpn.fire_transition

        def wrapped(*args, **kwargs):
            time.sleep(0.002)
            return original(*args, **kwargs)

        cpn.fire_transition = wrapped  # type: ignore[method-assign]

    slow_guard_net()
    rt = SimulationRuntime(cpn, marking, ctx)
    assert rt.submit_non_animated_batch(BatchConfig(mode="steps", max_steps=200))
    time.sleep(0.02)
    rt.request_stop()
    status = _wait_terminal(rt)
    assert status.terminal_reason == "stopped"
    assert status.firings < 200


def test_async_lease_expiry():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _chain_net(200)

    original = cpn.fire_transition

    def wrapped(*args, **kwargs):
        time.sleep(0.005)
        return original(*args, **kwargs)

    cpn.fire_transition = wrapped  # type: ignore[method-assign]

    rt = SimulationRuntime(cpn, marking, ctx)
    assert rt.submit_non_animated_batch(
        BatchConfig(mode="steps", max_steps=200),
        lease_s=0.05,
    )
    # Do not renew lease
    status = _wait_terminal(rt, timeout=3.0)
    assert status.terminal_reason == "lease_expired"


def test_async_monitor_hit():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _chain_net(10)
    mon = MonitorSpec(
        name="Clock0",
        slug=slugify_monitor_name("Clock0"),
        predicate=lambda c, m: True,
        before=True,
        transition_name="T1",
        default_enabled=True,
    )
    rt = SimulationRuntime(cpn, marking, ctx, monitors=[mon])
    assert rt.submit_non_animated_batch(BatchConfig(mode="steps", max_steps=10))
    status = _wait_terminal(rt)
    assert status.terminal_reason == "monitor"
    assert status.monitor_names == ("Clock0",)


def test_sync_blocks_while_async_owns():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig, StepResult

    cpn, marking, ctx = _chain_net(10)
    rt = SimulationRuntime(cpn, marking, ctx)
    hold = threading.Event()
    entered = threading.Event()
    original = rt._one_macro_step

    def blocked_step(config, *, stop_requested=None):
        entered.set()
        hold.wait(timeout=2.0)
        return StepResult(kind="done", firings=0)

    rt._one_macro_step = blocked_step  # type: ignore[method-assign]
    assert rt.submit_non_animated_batch(BatchConfig(mode="steps", max_steps=10))
    assert entered.wait(1.0)

    started = threading.Event()
    finished = threading.Event()

    def sync_fire():
        started.set()
        rt.fire("T1")
        finished.set()

    t = threading.Thread(target=sync_fire, daemon=True)
    t.start()
    assert started.wait(1.0)
    assert not finished.wait(0.1)
    hold.set()
    assert finished.wait(2.0)
    t.join(timeout=1.0)
    rt._one_macro_step = original  # type: ignore[method-assign]
    _wait_terminal(rt)


def test_poll_renews_lease():
    from cpnpy.visualization.streamlit.runtime import SimulationRuntime
    from cpnpy.visualization.streamlit.step_engine import BatchConfig

    cpn, marking, ctx = _chain_net(30)
    original = cpn.fire_transition

    def wrapped(*args, **kwargs):
        time.sleep(0.01)
        return original(*args, **kwargs)

    cpn.fire_transition = wrapped  # type: ignore[method-assign]
    rt = SimulationRuntime(cpn, marking, ctx)
    assert rt.submit_non_animated_batch(
        BatchConfig(mode="steps", max_steps=30),
        lease_s=0.15,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        st = rt.poll_status(renew=True)
        if not st.async_running:
            break
        time.sleep(0.05)
    status = rt.poll_status(renew=False)
    assert status.terminal_reason == "done"
    assert status.firings == 30
