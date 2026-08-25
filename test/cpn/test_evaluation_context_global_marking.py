import copy

import pytest

from cpnpy.cpn.cpn_imp import EvaluationContext, GlobalMarkingView, Marking


def test_global_marking_unbound_raises():
    context = EvaluationContext()
    context.install_global_marking()
    with pytest.raises(RuntimeError, match="active marking"):
        _ = context.env["global_marking"].time


def test_global_marking_reads_bound_clock():
    context = EvaluationContext()
    context.install_global_marking()
    marking = Marking()
    marking.global_clock = 42
    context.bind_marking(marking)
    assert context.env["global_marking"].time == 42


def test_global_marking_in_guard_namespace():
    context = EvaluationContext()
    context.install_global_marking()
    marking = Marking()
    marking.global_clock = 7
    context.bind_marking(marking)
    assert context.evaluate_guard("global_marking.time == 7", {})


def test_global_marking_in_action():
    context = EvaluationContext()
    context.install_global_marking()
    marking = Marking()
    marking.global_clock = 99
    context.bind_marking(marking)

    captured = {}

    def action(inp, out):
        captured["t"] = global_marking.time

    context.evaluate_action(action, {})
    assert captured["t"] == 99


def test_global_marking_tracks_deepcopied_marking():
    context = EvaluationContext()
    context.install_global_marking()
    original = Marking()
    original.global_clock = 1
    copy_marking = copy.deepcopy(original)
    copy_marking.global_clock = 50
    context.bind_marking(copy_marking)
    assert context.env["global_marking"].time == 50
    assert original.global_clock == 1


def test_context_deepcopy_clears_active_marking():
    context = EvaluationContext()
    context.install_global_marking()
    marking = Marking()
    marking.global_clock = 10
    context.bind_marking(marking)
    copied = copy.deepcopy(context)
    assert copied._active_marking is None
    assert isinstance(copied.env["global_marking"], GlobalMarkingView)
    with pytest.raises(RuntimeError, match="active marking"):
        _ = copied.env["global_marking"].time
