"""Shared Streamlit-free simulation step engine for sync and async batch paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from cpnpy.cpn.cpn_imp import CPN, EvaluationContext, Marking, format_exception_oneline
from cpnpy.simulation.simu import get_enabled_transitions
from cpnpy.visualization.streamlit.helpers import (
    MonitorSpec,
    batch_step_logic,
    enabled_transition_names,
    evaluate_monitors,
)


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for one Steps/Time batch run."""

    mode: str  # "steps" | "time"
    max_steps: int = 0
    target_time: int = 0
    auto_advance: bool = True


@dataclass
class StepResult:
    """Result of one fire or one batch macro-step."""

    kind: str
    firings: int = 0
    info: dict = field(default_factory=dict)
    monitors: list[str] = field(default_factory=list)
    error: str | None = None


def advance_clock(cpn: CPN, marking: Marking) -> bool:
    """Advance global clock once. Returns True if the clock moved."""
    before = marking.global_clock
    cpn.advance_global_clock(marking)
    return marking.global_clock != before


def fire_transition(
    *,
    cpn: CPN,
    marking: Marking,
    context: EvaluationContext,
    transition_name: str,
    monitors: list[MonitorSpec],
    enabled_slugs: frozenset[str],
    skip_monitors: bool = False,
) -> StepResult:
    """
    Fire one named transition with before/after monitor evaluation.

    Returns kind in {"fired", "monitor", "error"}.
    """
    try:
        trans = cpn.get_transition_by_name(transition_name)
        if trans is None:
            raise ValueError(f"Unknown transition {transition_name!r}.")
        all_enabled = get_enabled_transitions(
            cpn, marking, context, only_best_priority=False,
        )
        enabled_names = enabled_transition_names(all_enabled)
        before_hit: list[str] = []
        if not skip_monitors:
            before_hit = evaluate_monitors(
                monitors,
                phase="before",
                cpn=cpn,
                marking=marking,
                pending_transition=transition_name,
                enabled_names=enabled_names,
                enabled_slugs=enabled_slugs,
            )
        if before_hit:
            return StepResult(
                kind="monitor",
                firings=0,
                info={"monitor_hit": True, "monitors": before_hit, "in": [], "out": []},
                monitors=before_hit,
            )
        info = cpn.fire_transition(trans, marking, context)
        info["transition"] = transition_name
        after_hit: list[str] = []
        if not skip_monitors:
            after_hit = evaluate_monitors(
                monitors,
                phase="after",
                cpn=cpn,
                marking=marking,
                pending_transition=transition_name,
                enabled_names=enabled_names,
                enabled_slugs=enabled_slugs,
            )
        if after_hit:
            info["monitor_hit"] = True
            info["monitors"] = after_hit
            return StepResult(
                kind="monitor",
                firings=1,
                info=info,
                monitors=after_hit,
            )
        return StepResult(kind="fired", firings=1, info=info)
    except Exception as exc:
        return StepResult(
            kind="error",
            firings=0,
            info={"in": [], "out": []},
            error=format_exception_oneline(exc),
        )


def run_batch_macro_step(
    *,
    cpn: CPN,
    marking: Marking,
    context: EvaluationContext,
    monitors: list[MonitorSpec],
    enabled_slugs: frozenset[str],
    config: BatchConfig,
    batch_firings: int,
    stop_requested: bool,
    skip_monitors: bool = False,
) -> StepResult:
    """
    One batch macro-step: decide via batch_step_logic, then fire/advance/monitors.

    Returns kind in
    {"fired", "advanced", "done", "stopped", "deadlock", "idle", "monitor", "error"}.
    ``firings`` is the updated firing count after this step.
    """
    auto = True if config.mode == "time" else config.auto_advance
    enabled = get_enabled_transitions(cpn, marking, context)
    enabled_name = enabled[0].name if enabled else None

    decision = batch_step_logic(
        batch_mode=config.mode,
        batch_max_steps=int(config.max_steps),
        batch_target_time=int(config.target_time),
        batch_auto_advance=auto,
        batch_stop_requested=bool(stop_requested),
        batch_firings=int(batch_firings),
        global_clock=marking.global_clock,
        enabled_name=enabled_name,
    )

    if decision == "fired" and enabled_name:
        fire_result = fire_transition(
            cpn=cpn,
            marking=marking,
            context=context,
            transition_name=enabled_name,
            monitors=monitors,
            enabled_slugs=enabled_slugs,
            skip_monitors=skip_monitors,
        )
        if fire_result.kind == "error":
            return StepResult(
                kind="error",
                firings=batch_firings,
                info=fire_result.info,
                error=fire_result.error,
            )
        if fire_result.kind == "monitor":
            new_firings = batch_firings + (1 if fire_result.firings else 0)
            return StepResult(
                kind="monitor",
                firings=new_firings,
                info=fire_result.info,
                monitors=fire_result.monitors,
            )
        return StepResult(
            kind="fired",
            firings=batch_firings + 1,
            info=fire_result.info,
        )

    if decision == "need_advance":
        try:
            if not advance_clock(cpn, marking):
                return StepResult(kind="deadlock", firings=batch_firings)
            return StepResult(kind="advanced", firings=batch_firings)
        except Exception as exc:
            return StepResult(
                kind="error",
                firings=batch_firings,
                error=format_exception_oneline(exc),
            )

    # stopped | done | idle (and any other pure decision)
    return StepResult(kind=decision, firings=batch_firings)
