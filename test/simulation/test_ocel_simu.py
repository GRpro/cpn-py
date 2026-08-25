"""Regression: OCEL export must see action-produced output variables."""

import copy

from cpnpy.cpn.colorsets import IntegerColorSet
from cpnpy.cpn.cpn_imp import (
    Arc,
    CPN,
    EvaluationContext,
    Marking,
    Place,
    Transition,
)
from cpnpy.simulation.ocel_simu import simulate_cpn_to_ocel


def test_simulate_cpn_to_ocel_uses_action_output_vars():
    """Action sets out.rp0; output arc uses rp0 — must not NameError after fire."""
    cpn = CPN()
    p_init = Place("Init", IntegerColorSet())
    p_out = Place("Out", IntegerColorSet())

    def action_load(inp, out):
        out.rp0 = [10, 20]

    t = Transition(
        name="LoadPartitions",
        variables=["i"],
        action=action_load,
    )
    cpn.add_place(p_init)
    cpn.add_place(p_out)
    cpn.add_transition(t)
    cpn.add_arc(Arc(p_init, t, "i"))
    cpn.add_arc(Arc(t, p_out, "rp0"))

    marking = Marking()
    marking.set_tokens("Init", [0])
    context = EvaluationContext()

    ocel = simulate_cpn_to_ocel(cpn, marking, context)

    assert len(ocel.events) == 1
    assert ocel.events.iloc[0]["ocel:activity"] == "LoadPartitions"
    # Produced tokens appear as related objects (stringified values)
    oids = set(ocel.objects["ocel:oid"].astype(str))
    assert "10" in oids
    assert "20" in oids


def test_fire_transition_returns_binding_after_action():
    cpn = CPN()
    p_init = Place("Init", IntegerColorSet())
    p_out = Place("Out", IntegerColorSet())

    def action_load(inp, out):
        out.rp0 = [7]

    t = Transition(name="Load", variables=["i"], action=action_load)
    cpn.add_place(p_init)
    cpn.add_place(p_out)
    cpn.add_transition(t)
    cpn.add_arc(Arc(p_init, t, "i"))
    cpn.add_arc(Arc(t, p_out, "rp0"))

    marking = Marking()
    marking.set_tokens("Init", [1])
    context = EvaluationContext()
    binding = cpn._find_binding(t, marking, context)
    info = cpn.fire_transition(t, marking, context, binding)

    assert "binding_after_action" in info
    assert info["binding_after_action"]["i"] == 1
    assert info["binding_after_action"]["rp0"] == [7]


def test_ocel_simu_global_marking_tracks_deepcopied_clock():
    """After deepcopy(marking), global_marking.time must follow the sim copy."""
    cpn = CPN()
    p_init = Place("Init", IntegerColorSet())
    p_out = Place("Out", IntegerColorSet())

    def action_stamp(inp, out):
        out.t = global_marking.time

    t = Transition(name="Stamp", variables=["i"], action=action_stamp)
    cpn.add_place(p_init)
    cpn.add_place(p_out)
    cpn.add_transition(t)
    cpn.add_arc(Arc(p_init, t, "i"))
    cpn.add_arc(Arc(t, p_out, "t"))

    initial = Marking()
    initial.set_tokens("Init", [0])
    initial.global_clock = 0

    context = EvaluationContext()
    context.install_global_marking()

    simulate_cpn_to_ocel(cpn, initial, context)

    # Engine deepcopies marking; bound global_marking must read the copy's clock.
    assert context.env["global_marking"].time == initial.global_clock
    # If closure captured build-time marking at 0 only, copy could diverge;
    # after one fire the sim copy's clock stays 0 here — prove bind uses copy identity.
    copy_marking = copy.deepcopy(initial)
    copy_marking.global_clock = 123
    context.bind_marking(copy_marking)
    assert context.env["global_marking"].time == 123
    assert initial.global_clock == 0

    captured = {}

    def action_read(inp, out):
        captured["t"] = global_marking.time

    copy_marking.global_clock = 456
    context.bind_marking(copy_marking)
    context.evaluate_action(action_read, {})
    assert captured["t"] == 456
