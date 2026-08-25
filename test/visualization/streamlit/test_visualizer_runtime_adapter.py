"""Adapter tests for CPNStreamlitVisualizer runtime cache and marking ownership."""

from types import SimpleNamespace
from unittest.mock import MagicMock

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
from cpnpy.visualization.streamlit.runtime import SimulationRuntime
from cpnpy.visualization.streamlit.step_engine import BatchConfig


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


class _Session(dict):
    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


@pytest.fixture
def st_session(monkeypatch):
    session = _Session()
    fake_st = SimpleNamespace(
        session_state=session,
        rerun=MagicMock(),
        button=MagicMock(return_value=False),
        text=MagicMock(),
        caption=MagicMock(),
        markdown=MagicMock(),
        number_input=MagicMock(return_value=500),
        checkbox=MagicMock(return_value=False),
        radio=MagicMock(return_value="Steps"),
        selectbox=MagicMock(return_value="Force"),
        slider=MagicMock(return_value=100),
        download_button=MagicMock(),
        file_uploader=MagicMock(return_value=None),
        expander=MagicMock(),
        sidebar=MagicMock(),
        toast=MagicMock(),
    )
    # expander / sidebar as context managers
    fake_st.expander.return_value.__enter__ = lambda s: s
    fake_st.expander.return_value.__exit__ = lambda *a: None
    fake_st.sidebar.__enter__ = lambda s: s
    fake_st.sidebar.__exit__ = lambda *a: None

    import cpnpy.visualization.streamlit.ui as viz_mod

    monkeypatch.setattr(viz_mod, "st", fake_st)
    return session, fake_st, viz_mod


def test_triple_path_caches_same_runtime(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v1 = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="rt_cache",
    )
    v2 = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="rt_cache",
    )
    assert v1.runtime is v2.runtime
    assert session["rt_cache"] is v1.runtime
    assert isinstance(session["rt_cache"], SimulationRuntime)


def test_runtime_kwarg_path_uses_provided_instance(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    rt = SimulationRuntime(cpn, marking, ctx)
    v = viz_mod.CPNStreamlitVisualizer(runtime=rt, session_key="rt_explicit")
    assert v.runtime is rt
    assert session["rt_explicit"] is rt


def test_marking_property_is_runtime_marking(st_session):
    _session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="rt_mark",
    )
    assert v.marking is v.runtime.marking
    assert id(v.marking) == id(marking)


def test_graph_marking_frozen_while_async_flag(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="rt_freeze",
    )
    import copy

    frozen = copy.deepcopy(v.marking)
    session[v._k("graph_freeze_marking")] = frozen
    session[v._k("async_batch_active")] = True
    v.runtime.fire("T1")
    assert len(v.marking.get_multiset("P1").tokens) == 2
    assert len(v._graph_marking().get_multiset("P1").tokens) == 3


def test_async_start_stop_poll_terminal(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net(tokens=list(range(8)))
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="rt_async",
    )
    v._init_session_defaults()
    v._start_batch("steps", 8, 0, False, True, 500)
    assert session[v._k("async_batch_active")] is True
    assert session[v._k("graph_freeze_marking")] is not None

    assert v.runtime.wait_until_idle(timeout=2.0)
    # Avoid 3s poll sleep; worker already finished — drain terminal state.
    viz_mod.time.sleep = lambda _s: None
    v._run_fast_batch_after_sidebar()

    assert not session.get(v._k("batch_running"))
    assert not session.get(v._k("async_batch_active"))
    assert session.get(v._k("graph_freeze_marking")) is None
    assert len(v.runtime.marking.get_multiset("P2").tokens) == 8


def test_async_poll_keeps_batch_status_running_for_repair(st_session):
    """While async_running, poll must keep batch_status exactly 'Running' for repair."""
    from cpnpy.visualization.streamlit.runtime import RuntimeStatus

    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net(tokens=list(range(8)))
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="rt_repair",
    )
    v._init_session_defaults()
    v._start_batch("steps", 8, 0, False, True, 500)
    # Stop the real worker so the test is deterministic.
    v.runtime.request_stop()
    assert v.runtime.wait_until_idle(timeout=2.0)
    # Re-arm UI flags as if a batch were still async-running.
    session[v._k("batch_running")] = True
    session[v._k("batch_phase")] = "fast"
    session[v._k("batch_status")] = "Running"
    session[v._k("async_batch_active")] = True
    session[v._k("graph_freeze_marking")] = marking

    def fake_poll(*, renew=True):
        return RuntimeStatus(
            phase="running",
            status="Running · 3 firings · clock 0",
            global_clock=0,
            firings=3,
            stop_requested=False,
            lease_deadline=None,
            terminal_reason=None,
            monitor_names=(),
            error=None,
            async_running=True,
        )

    v.runtime.poll_status = fake_poll  # type: ignore[method-assign]
    viz_mod.time.sleep = lambda _s: None
    v._run_fast_batch_after_sidebar()
    assert session[v._k("batch_status")] == "Running"
    assert session[v._k("batch_firings")] == 3

    v._repair_batch_session()
    assert session[v._k("batch_running")] is True
    assert session[v._k("async_batch_active")] is True
    assert session[v._k("batch_status")] == "Running"

    # Contrast: writing the decorated string would make repair clear running.
    session[v._k("batch_status")] = "Running · 3 firings · clock 0"
    v._repair_batch_session()
    assert session[v._k("batch_running")] is False
    assert session.get(v._k("async_batch_active")) is None


def test_monitor_resume_skip_once_resubmit(st_session):
    _session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="rt_mon",
    )
    v.register_monitor(
        "Hit",
        lambda c, m: True,
        before=True,
        transition_name="T1",
    )
    v._sync_runtime_monitors()
    status = v.runtime.run_batch_sync(BatchConfig(mode="steps", max_steps=5))
    assert status.terminal_reason == "monitor"
    v.runtime.skip_monitors_once()
    status = v.runtime.run_batch_sync(BatchConfig(mode="steps", max_steps=5))
    assert status.firings >= 1


def test_layout_export_empty_session_uses_capture_button(st_session):
    session, fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="layout_export_empty",
    )
    v._init_session_defaults()
    v._notify = MagicMock()  # type: ignore[method-assign]
    fake_st.button.return_value = False

    v._render_layout_export_controls(batch_running=False)

    fake_st.download_button.assert_not_called()
    fake_st.button.assert_called_once()
    assert fake_st.button.call_args.kwargs["key"] == v._k("export_graph_layout_capture")
    assert session.get(v._k("graph_layout_positions")) in (None, {})


def test_layout_export_filled_session_uses_download(st_session):
    session, fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="layout_export_full",
    )
    v._init_session_defaults()
    node_ids = v._graph_node_ids()
    positions = {nid: {"x": float(i), "y": 1.0} for i, nid in enumerate(node_ids)}
    v._set_saved_layout_session(positions)

    v._render_layout_export_controls(batch_running=False)

    fake_st.download_button.assert_called_once()
    data = fake_st.download_button.call_args.kwargs["data"]
    assert '"positions"' in data
    assert node_ids[0] in data
    fake_st.button.assert_not_called()


def test_prepare_data_auto_sync_once_when_session_empty(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="layout_auto_sync",
    )
    v._init_session_defaults()

    first = v._prepare_data([])
    assert first.get("sync_layout_to_session") is True
    assert session[v._k("layout_auto_sync_attempted")] is True

    second = v._prepare_data([])
    assert "sync_layout_to_session" not in second


def test_prepare_data_force_sync_from_export_click(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="layout_force_sync",
    )
    v._init_session_defaults()
    session[v._k("layout_auto_sync_attempted")] = True
    v._request_layout_session_sync()

    payload = v._prepare_data([])
    assert payload.get("sync_layout_to_session") is True
    assert v._k("sync_layout_to_session") not in session


def test_handle_graph_layout_sync_fills_session_for_export(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="layout_sync_fill",
    )
    v._init_session_defaults()
    node_ids = v._graph_node_ids()
    positions = {nid: {"x": 10.0, "y": 20.0} for nid in node_ids}

    v._handle_graph_layout_sync({"positions": positions, "view": {"scale": 1.0}})

    assert v._session_has_layout_positions()
    export = v._layout_export_json()
    assert node_ids[0] in export
    assert session[v._k("graph_layout_view")]["scale"] == 1.0


def test_layout_sync_empty_to_filled_requests_rerun(st_session, monkeypatch):
    """Sidebar Export renders before the graph; empty→filled must schedule a rerun."""
    import json

    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="layout_sync_rerun",
    )
    v._init_session_defaults()
    node_ids = v._graph_node_ids()
    positions = {nid: {"x": 1.0, "y": 2.0} for nid in node_ids}
    layout_raw = json.dumps({
        "type": "layout",
        "positions": positions,
        "view": {"scale": 1.0},
    })
    v._request_rerun = MagicMock()  # type: ignore[method-assign]
    v._handle_graph_component_pick = MagicMock()  # type: ignore[method-assign]
    monkeypatch.setattr(viz_mod, "cpn_graph", MagicMock(return_value=layout_raw))

    v._render_graph_panel(height=400, enabled_names=[], last_fired={})
    v._request_rerun.assert_called_once()
    assert v._session_has_layout_positions()

    v._request_rerun.reset_mock()
    v._render_graph_panel(height=400, enabled_names=[], last_fired={})
    v._request_rerun.assert_not_called()
    assert session[v._k("graph_layout_positions")]


def test_sync_guard_error_status_messages_notify_and_clear(st_session):
    session, _fake_st, viz_mod = st_session
    cpn, marking, ctx = _simple_net()
    v = viz_mod.CPNStreamlitVisualizer(
        cpn, marking, context=ctx, session_key="guard_err_sync",
    )
    v._init_session_defaults()
    ctx.record_guard_error("T1", ValueError("bad guard"))
    ctx.record_guard_error("T2", RuntimeError("boom"))

    v._sync_guard_error_status_messages()

    store = session[v._k("status_messages")]
    assert store["guard_error:T1"]["message"] == (
        "Guard evaluation failed for T1 — ValueError: bad guard"
    )
    assert store["guard_error:T2"]["level"] == "error"

    ctx.clear_guard_errors()
    ctx.record_guard_error("T1", ValueError("still bad"))
    v._sync_guard_error_status_messages()

    assert "guard_error:T2" not in store
    assert "still bad" in store["guard_error:T1"]["message"]
