"""Simulation runtime: owns CPN + Marking + EvaluationContext; sync and async batch."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Callable

from cpnpy.cpn.cpn_imp import CPN, EvaluationContext, Marking
from cpnpy.visualization.streamlit.step_engine import (
    BatchConfig,
    StepResult,
    advance_clock as engine_advance_clock,
    fire_transition as engine_fire,
    run_batch_macro_step,
)
from cpnpy.visualization.streamlit.helpers import (
    FAST_BATCH_MAX_ITERATIONS,
    MonitorSpec,
    monitor_batch_status_message,
)


DEFAULT_LEASE_S = 15.0


@dataclass(frozen=True)
class RuntimeStatus:
    """Immutable status snapshot for UI / callers (no token map)."""

    phase: str  # idle | running | terminal
    status: str
    global_clock: int
    firings: int
    stop_requested: bool
    lease_deadline: float | None
    terminal_reason: str | None
    monitor_names: tuple[str, ...]
    error: str | None
    async_running: bool


def _status_message(
    reason: str,
    *,
    firings: int,
    clock: int,
    mode: str,
    monitors: tuple[str, ...] = (),
    error: str | None = None,
) -> str:
    if reason == "stopped":
        return "Stopped"
    if reason == "lease_expired":
        return "Stopped (lease expired)"
    if reason == "error":
        return f"Error — {error or 'unknown error'}"
    if reason == "deadlock":
        return f"Deadlock at time {clock}"
    if reason == "idle":
        return f"Finished (idle at time {clock})"
    if reason == "monitor":
        return monitor_batch_status_message(list(monitors))
    if reason == "done":
        if mode == "time":
            return f"Finished (time target @ {clock})"
        return f"Finished ({firings} transitions)"
    return "Finished"


class SimulationRuntime:
    """
    Owns the live Marking and EvaluationContext for a simulation run.

    Public API is Streamlit-free: hold a normal Python reference. Async
    non-animated Steps/Time batches run on a daemon worker; other mutations
    serialize behind the same ownership lock.
    """

    def __init__(
        self,
        cpn: CPN,
        marking: Marking,
        context: EvaluationContext | None = None,
        monitors: list[MonitorSpec] | None = None,
    ):
        self._cpn = cpn
        self._marking = marking
        self._context = context or EvaluationContext()
        self._monitors: list[MonitorSpec] = list(monitors or [])
        self._enabled_slugs: frozenset[str] = frozenset(m.slug for m in self._monitors)
        self._initial_marking = copy.deepcopy(marking)

        self._owner = threading.RLock()
        self._gate = threading.Condition(self._owner)
        self._status_lock = threading.Lock()
        self._stop = threading.Event()
        self._skip_monitors = False
        self._worker: threading.Thread | None = None
        self._lease_deadline: float | None = None
        self._lease_s = DEFAULT_LEASE_S
        self._async_running = False
        self._firings = 0
        self._published_clock = marking.global_clock
        self._batch_mode = "steps"
        self._terminal_reason: str | None = None
        self._monitor_names: tuple[str, ...] = ()
        self._error: str | None = None
        self._status_text = "Idle"
        self._phase = "idle"

    def _await_sync_turn(self) -> None:
        """Wait until async batch releases ownership. Caller must hold ``_gate``."""
        while self._async_running:
            self._gate.wait()

    # --- owned state ---------------------------------------------------------

    @property
    def cpn(self) -> CPN:
        return self._cpn

    @property
    def marking(self) -> Marking:
        return self._marking

    @property
    def context(self) -> EvaluationContext:
        return self._context

    def set_monitors(
        self,
        monitors: list[MonitorSpec],
        enabled_slugs: frozenset[str] | None = None,
    ) -> None:
        with self._gate:
            self._await_sync_turn()
            self._monitors = list(monitors)
            if enabled_slugs is None:
                self._enabled_slugs = frozenset(m.slug for m in self._monitors)
            else:
                self._enabled_slugs = frozenset(enabled_slugs)

    def skip_monitors_once(self) -> None:
        self._skip_monitors = True

    # --- status --------------------------------------------------------------

    def poll_status(self, *, renew: bool = True) -> RuntimeStatus:
        with self._status_lock:
            if renew and self._async_running and self._lease_deadline is not None:
                self._lease_deadline = time.monotonic() + self._lease_s
            return RuntimeStatus(
                phase=self._phase,
                status=self._status_text,
                global_clock=self._published_clock,
                firings=self._firings,
                stop_requested=self._stop.is_set(),
                lease_deadline=self._lease_deadline,
                terminal_reason=self._terminal_reason,
                monitor_names=self._monitor_names,
                error=self._error,
                async_running=self._async_running,
            )

    def renew_lease(self, lease_s: float | None = None) -> None:
        with self._status_lock:
            if not self._async_running:
                return
            seconds = self._lease_s if lease_s is None else float(lease_s)
            self._lease_s = seconds
            self._lease_deadline = time.monotonic() + seconds

    def request_stop(self) -> None:
        self._stop.set()

    # --- sync mutations ------------------------------------------------------

    def fire(self, transition_name: str) -> StepResult:
        with self._gate:
            self._await_sync_turn()
            skip = self._consume_skip_monitors()
            result = engine_fire(
                cpn=self._cpn,
                marking=self._marking,
                context=self._context,
                transition_name=transition_name,
                monitors=self._monitors,
                enabled_slugs=self._enabled_slugs,
                skip_monitors=skip,
            )
            if result.kind == "fired":
                self._firings += 1
            elif result.kind == "monitor" and result.firings:
                self._firings += 1
            with self._status_lock:
                self._published_clock = self._marking.global_clock
            return result

    def advance_clock(self) -> bool:
        with self._gate:
            self._await_sync_turn()
            moved = engine_advance_clock(self._cpn, self._marking)
            with self._status_lock:
                self._published_clock = self._marking.global_clock
            return moved

    def run_macro_step(
        self,
        config: BatchConfig,
        *,
        stop_requested: bool = False,
        batch_firings: int | None = None,
    ) -> StepResult:
        """One sync batch macro-step (animated / caller-driven loops)."""
        with self._gate:
            self._await_sync_turn()
            if batch_firings is not None:
                self._firings = int(batch_firings)
            return self._one_macro_step(config, stop_requested=stop_requested)

    def _one_macro_step(
        self,
        config: BatchConfig,
        *,
        stop_requested: bool | None = None,
    ) -> StepResult:
        skip = self._consume_skip_monitors()
        stop = self._stop.is_set() if stop_requested is None else bool(stop_requested)
        result = run_batch_macro_step(
            cpn=self._cpn,
            marking=self._marking,
            context=self._context,
            monitors=self._monitors,
            enabled_slugs=self._enabled_slugs,
            config=config,
            batch_firings=self._firings,
            stop_requested=stop,
            skip_monitors=skip,
        )
        self._firings = result.firings
        with self._status_lock:
            self._published_clock = self._marking.global_clock
            self._status_text = (
                f"Running · {self._firings} firings · clock {self._published_clock}"
            )
        return result

    def reset_marking(self, marking: Marking | None = None) -> None:
        with self._gate:
            if self._async_running:
                self._stop.set()
            self._await_sync_turn()
            source = marking if marking is not None else self._initial_marking
            restored = copy.deepcopy(source)
            # Mutate the owned marking object in place so callers keep identity.
            self._marking._marking.clear()
            for place_name, ms in restored._marking.items():
                self._marking._marking[place_name] = ms
            self._marking.global_clock = restored.global_clock
            if marking is not None:
                self._initial_marking = copy.deepcopy(marking)
            self._firings = 0
            self._published_clock = self._marking.global_clock
            self._terminal_reason = None
            self._monitor_names = ()
            self._error = None
            self._status_text = "Idle"
            self._phase = "idle"
            self._stop.clear()

    def run_batch_sync(
        self,
        config: BatchConfig,
        *,
        max_iterations: int = FAST_BATCH_MAX_ITERATIONS,
        should_stop: Callable[[], bool] | None = None,
    ) -> RuntimeStatus:
        with self._gate:
            self._await_sync_turn()
            self._prepare_batch(config)
            iterations = 0
            while iterations < max_iterations:
                if should_stop and should_stop():
                    self._stop.set()
                result = self._one_macro_step(config)
                iterations += 1
                if result.kind in {
                    "stopped", "done", "deadlock", "idle", "error", "monitor",
                }:
                    self._publish_terminal(result, config.mode)
                    break
            else:
                self._publish_terminal(
                    StepResult(kind="done", firings=self._firings),
                    config.mode,
                    override_reason="done",
                    override_status="Finished (iteration cap — partial)",
                )
            return self.poll_status(renew=False)

    # --- async ---------------------------------------------------------------

    def submit_non_animated_batch(
        self,
        config: BatchConfig,
        *,
        lease_s: float = DEFAULT_LEASE_S,
        max_iterations: int = FAST_BATCH_MAX_ITERATIONS,
    ) -> bool:
        """Start a daemon worker for non-animated Steps/Time. Returns False if busy."""
        with self._gate:
            if self._async_running:
                return False
            self._prepare_batch(config)
            self._lease_s = float(lease_s)
            self._lease_deadline = time.monotonic() + self._lease_s
            self._async_running = True
            self._phase = "running"
            self._status_text = "Running"
            self._terminal_reason = None
            self._monitor_names = ()
            self._error = None

        def worker() -> None:
            with self._gate:
                try:
                    self._async_worker_loop(
                        config, max_iterations=max_iterations,
                    )
                finally:
                    with self._status_lock:
                        self._async_running = False
                        self._lease_deadline = None
                        if self._phase == "running":
                            self._phase = "terminal"
                    self._gate.notify_all()

        self._worker = threading.Thread(
            target=worker, name="cpn-sim-runtime", daemon=True,
        )
        self._worker.start()
        return True

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        t = self._worker
        if t is None:
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

    # --- internals -----------------------------------------------------------

    def _prepare_batch(self, config: BatchConfig) -> None:
        self._stop.clear()
        self._batch_mode = config.mode
        self._firings = 0
        self._terminal_reason = None
        self._monitor_names = ()
        self._error = None

    def _consume_skip_monitors(self) -> bool:
        if self._skip_monitors:
            self._skip_monitors = False
            return True
        return False

    def _publish_terminal(
        self,
        result: StepResult,
        mode: str,
        *,
        override_reason: str | None = None,
        override_status: str | None = None,
    ) -> None:
        reason = override_reason or result.kind
        monitors = tuple(result.monitors)
        error = result.error
        msg = override_status or _status_message(
            reason,
            firings=result.firings,
            clock=self._marking.global_clock,
            mode=mode,
            monitors=monitors,
            error=error,
        )
        with self._status_lock:
            self._terminal_reason = reason
            self._monitor_names = monitors
            self._error = error
            self._status_text = msg
            self._phase = "terminal"
            self._firings = result.firings
            self._published_clock = self._marking.global_clock

    def _async_worker_loop(self, config: BatchConfig, *, max_iterations: int) -> None:
        """Caller must already hold ``_owner`` for transferable ownership."""
        iterations = 0
        while iterations < max_iterations:
            with self._status_lock:
                deadline = self._lease_deadline
            if deadline is not None and time.monotonic() >= deadline:
                self._publish_terminal(
                    StepResult(kind="stopped", firings=self._firings),
                    config.mode,
                    override_reason="lease_expired",
                )
                return

            result = self._one_macro_step(config)
            iterations += 1
            if result.kind in {
                "stopped", "done", "deadlock", "idle", "error", "monitor",
            }:
                self._publish_terminal(result, config.mode)
                return

        self._publish_terminal(
            StepResult(kind="done", firings=self._firings),
            config.mode,
            override_reason="done",
            override_status="Finished (iteration cap — partial)",
        )
