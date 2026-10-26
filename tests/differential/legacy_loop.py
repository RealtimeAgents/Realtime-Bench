"""The frame-driven decision loop runner.main() ran for the wall and token clocks before the event
kernel's shell (realtime_framework/evaluation/runner.py run_stream), moved here unchanged and kept
only to be run side by side with it.

Each token-clock decision follows one protocol: capture a frame, prefill and generate, advance
the world to the selected action's clock position, apply the action with time frozen, then advance
to the next scheduled observation boundary. For the finite token clock,
``B=tokens_per_frame`` is a token-to-time rate, never a per-decision output cap. An action after
``n`` provider-reported output tokens lands ``n/B`` frames after its observation. A response may
cross any number of frame boundaries; an action whose completed response lands beyond the episode
horizon is not applied. With unlimited B, output length adds no action delay: the action lands at
the observed tick and the world then advances one frame before the next observation.

Frames are the model-input frames, ``frame_ticks`` environment ticks apart: frame j is due at
j * frame_ticks, and one that falls between two ticks is observed at the next tick. The episode
horizon ``max_ticks`` is in environment ticks.

`env` must provide: capture() -> Observation, execute(action) -> status str, tick_index()
(current env tick), done() -> bool, wait_until_tick(t). Optional: render_current_frame() commits
action-visible pixels without advancing logical time; time_remaining() and past_deadline() enforce
REAL timeouts. Δ (tick seconds) sets the wall-clock<->tick mapping.

Runtime invariants:
  * Operational timeouts: wall mode is bounded by the per-move/deadline wall clock. Token
    mode uses only a provider-health watchdog; hitting it invalidates the run instead of affecting
    score. Its per-move cap is applied later in token-derived simulated seconds.
  * Pre-dispatch validation: the deliberation budget is checked and the
    deadline is rechecked before execution; over-budget or late actions are skipped.
  * Always pace, execute conditionally: wait_until_tick(τ_k) runs every iteration even when
    there is no action, so an empty/failed/timed-out decision can't hot-loop requests in one tick.
  * Action lifecycle: the runner owns proposed -> {applied | skipped_timeout |
    skipped_budget | not_applied_time_limit | not_applied_tick_limit | terminal |
    not_applied_terminal | rejected | terminal_ignored | wait | none | invalid_run} and calls
    adapter.commit_action(status).
  * Latched outcome: the stop reason and the termination record (its end time by rule and the
    snapshot's time) are frozen on env at loop exit, also when the loop raises, so post-loop cleanup
    crossing the deadline can't rewrite a real environment terminal state into time_limit or
    lengthen the episode.
"""

from __future__ import annotations

import math
import os
import traceback

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.runner import RUNNER_TIMING
from realtime_framework.sandbox.actions import ENVIRONMENT_BUDGET
from realtime_framework.sandbox.budget import operation_budget

# Small settle margin added to the wall-clock bound so a well-behaved call that is about to return
# at the deadline can finish; per-move compute caps receive no grace period. The kernel has none
# (K-a), so the runner's timing no longer carries it and the legacy loop reads it itself.
DEADLINE_GRACE_S = float(os.environ.get("STREAM_DEADLINE_GRACE_S", "2.0"))
# Bound on waiting out per-move workers a timeout abandoned, before teardown/exit (_drain_abandoned).
# Sized for one native generate to finish returning, not for a wedged call to recover.
ABANDONED_DRAIN_S = RUNNER_TIMING.abandoned_drain_s
# A stream is frozen only when all three producer-owned identity signals repeat while real time
# advances.  Screenshot equality by itself is valid for menus/static scenes, and equal tick values
# can occur when capture is faster than the environment clock, so neither is sufficient alone.
OBSERVATION_FREEZE_N = RUNNER_TIMING.observation_freeze_n
# A token-clock call has no wall-clock scoring deadline by design, but it still needs an operational
# watchdog so a dead provider cannot hang a lane forever. Hitting this bound invalidates the run;
# it is never treated as model failure or converted into simulated time.
PROVIDER_WATCHDOG_S = RUNNER_TIMING.provider_watchdog_s


class ObservationStreamWatchdog:
    """Detect a producer replaying the exact same observation instead of advancing its stream."""

    def __init__(self, threshold=OBSERVATION_FREEZE_N, *, clock):
        self.threshold = max(2, int(threshold))
        self.clock = clock
        self._fingerprint = None
        self._first_seen_at = None
        self._count = 0

    fingerprint = staticmethod(episode_records.observation_fingerprint)

    def observe(self, observation):
        """Return True once N identical identities arrive with a progressing wall clock."""
        now = self.clock()
        fingerprint = self.fingerprint(observation)
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._first_seen_at = now
            self._count = 1
            return False
        self._count += 1
        first_seen_at = self._first_seen_at
        return self._count >= self.threshold and first_seen_at is not None and now > first_seen_at


def _whole_up(value):
    """The smallest whole number at or above ``value``, within float noise."""
    rounded = round(value)
    return int(rounded) if abs(value - rounded) < 1e-9 else math.ceil(value)


def _run_stream_legacy(
    adapter,
    env,
    delta_s=1.0,
    frame_ticks=1.0,
    max_ticks=60,
    log=print,
    think_budget_s=None,
    per_move_cap_s=None,
    clock_mode="token",
    tokens_per_frame=None,
    trace=None,
    clock=None,
):
    """The turn loop main() ran before the kernel's shell, kept only to be run side by side with
    it. ``clock`` is the episode's operational clock (realtime_framework/clock.py), real time when
    omitted; on the wall clock it is also the semantic clock ``env`` reads."""
    clock = clock or SystemClock()
    if clock_mode not in {"token", "wall"}:
        raise ValueError(f"unsupported clock mode: {clock_mode!r}")
    if delta_s <= 0:
        raise ValueError("delta_s must be positive")
    if frame_ticks <= 0:
        raise ValueError("frame_ticks must be positive")
    if tokens_per_frame is not None and (
        type(tokens_per_frame) is not int or tokens_per_frame <= 0
    ):
        raise ValueError("tokens_per_frame must be a positive integer or None")
    horizon_setter = getattr(env, "set_episode_horizon", None)
    if clock_mode == "token":
        if callable(horizon_setter):
            horizon_setter(max_ticks)
        else:
            setattr(env, "_episode_horizon_tick", float(max_ticks))
    setattr(adapter, "_clock_mode", clock_mode)
    timeline = []  # (k, t_k, delta_ticks, tau_k, action)
    abandoned = []  # workers a timeout left running; drained at teardown (see _drain_abandoned)
    _drained = []  # one-shot latch for _drain_abandoned
    tau_prev = -1
    spent = 0.0  # cumulative deliberation (model-inference) time -- the "clock"
    observation_watchdog = ObservationStreamWatchdog(clock=clock.monotonic)
    trace_failed = [False]
    # When the loop first learned of each kind of end, in world seconds since the episode began.
    end_signals = {}

    def _world_s():
        position = getattr(env, "clock_position", None)
        return float(position() if callable(position) else env.tick_index()) * delta_s

    def _emit(event, payload=None):
        """Best-effort call surface whose failure invalidates, but never rewrites, runtime state."""
        if trace is None or trace_failed[0]:
            return None
        try:
            return trace.emit(event, payload or {})
        except Exception:
            trace_failed[0] = True
            failure = traceback.format_exc()
            end_signals.setdefault("invalid_run", _world_s())
            if getattr(env, "_invalid_run", None) is None:
                setattr(
                    env,
                    "_invalid_run",
                    {
                        "category": "harness",
                        "reason": "episode_trace_write_failed",
                        "detail": failure[-2000:],
                    },
                )
            log(f"[invalid-run] harness:episode_trace_write_failed — {failure}")
            return None

    def _checkpoint_trace():
        if trace is None or trace_failed[0]:
            return
        checkpoint = getattr(trace, "checkpoint", None)
        if checkpoint is None:
            return
        try:
            checkpoint()
        except Exception:
            trace_failed[0] = True
            _invalidate("harness", "episode_trace_checkpoint_failed", traceback.format_exc())

    def _invalidate(category, reason, detail=None):
        """Latch the first non-model failure. Invalid runs are never scored by the runner."""
        end_signals.setdefault("invalid_run", _world_s())
        if getattr(env, "_invalid_run", None) is None:
            value = {"category": category, "reason": reason}
            if detail:
                # The end of a traceback names the exception.
                value["detail"] = str(detail)[-2000:]
            setattr(env, "_invalid_run", value)
            log(f"[invalid-run] {category}:{reason}" + (f" — {detail}" if detail else ""))
            _emit(TraceEvent.RUN_INVALIDATED, value)
        return getattr(env, "_invalid_run")

    def _frame_at(tick):
        """The last model-input frame due at or before ``tick``."""
        return math.floor(tick / frame_ticks + 1e-9)

    def _accept_observation(obs):
        """Gate captured frames through the replay watchdog; a frozen stream invalidates the run."""
        if not observation_watchdog.observe(obs):
            return True
        fingerprint = ObservationStreamWatchdog.fingerprint(obs)
        freezes = int(getattr(env, "_stream_freeze_count", 0) or 0) + 1
        setattr(env, "_stream_freeze_count", freezes)
        setattr(env, "_stream_frozen", True)
        log(
            f"[default] observation stream FROZEN (same id/tick/hash x"
            f"{OBSERVATION_FREEZE_N}); invalidating run"
        )
        _invalidate("environment", "observation_stream_frozen", repr(fingerprint))
        # abort() releases the wedged provider session before the invalid run ends.
        abort = getattr(adapter, "abort", None)
        if abort is not None:
            try:
                abort()
            except Exception:
                pass
        return False

    # ---- env capability shims (optional deadline hooks) ----
    def _time_remaining():
        f = getattr(env, "time_remaining", None)
        if f is None:
            return None
        try:
            value = f()
            # Token-clock environments intentionally have no wall-clock scoring deadline.  They
            # expose the shared optional hook but return None so _move_timeout() can fall back to
            # the provider-health watchdog without turning that sentinel into an invalid run.
            return None if value is None else float(value)
        except Exception:
            _invalidate("environment", "time_remaining_failed", traceback.format_exc())
            return None

    def _past_deadline():
        f = getattr(env, "past_deadline", None)
        if f is None:
            return False
        try:
            return bool(f())
        except Exception:
            _invalidate("environment", "deadline_check_failed", traceback.format_exc())
            return True

    def _episode_over():
        """Recheck terminal state immediately before dispatching an action.

        Inference may span the episode deadline. Preserve the terminal latch
        set by ``env.done()`` so finalization reports the same outcome."""
        f = getattr(env, "done", None)
        if f is None:
            return False
        queried_at = _world_s()
        try:
            over = bool(f())
        except Exception:
            _invalidate("environment", "terminal_state_check_failed", traceback.format_exc())
            return True
        return _end_reported(over, queried_at)

    def _end_reported(over, queried_at):
        """Record the ends a done() query reported at the time the query was made."""
        if over and getattr(env, "_invalid_run", None):
            end_signals.setdefault("invalid_run", queried_at)
        if over and getattr(env, "_episode_over", False):
            end_signals.setdefault("environment", queried_at)
        return over

    def _move_timeout():
        """Return the tightest operational wall timeout and its attribution."""
        bounds = []
        # In token mode this cap is evaluated against token-derived simulated seconds after the
        # response. Using it as a wall timeout would reintroduce hardware/provider speed bias.
        if per_move_cap_s is not None and clock_mode != "token":
            bounds.append((max(0.1, per_move_cap_s), "per_move_cap"))
        tr = _time_remaining()
        if tr is not None:
            bounds.append((max(0.1, tr + DEADLINE_GRACE_S), "wall_deadline"))
        if clock_mode == "token" and PROVIDER_WATCHDOG_S > 0:
            bounds.append((PROVIDER_WATCHDOG_S, "provider_watchdog"))
        return min(bounds) if bounds else (None, None)

    def _bounded_call(fn, timeout):
        """Run ``fn`` in a daemon worker with a bounded wait.

        Return ``(result, kind, error)`` with kind ``ok``, ``timeout``, or ``error``.
        On timeout, request adapter cancellation and retain the worker for teardown."""
        if timeout is None:
            try:
                return fn(), "ok", None
            except Exception:
                return None, "error", traceback.format_exc()
        box = {}
        done = clock.event()

        def worker():
            try:
                box["r"] = fn()
            except Exception:
                box["e"] = traceback.format_exc()
            finally:
                done.set()

        worker_thread = clock.start_thread(worker)
        if not clock.wait(done, timeout):
            abandoned.append((worker_thread, done))  # still running; drained at teardown
            ab = getattr(adapter, "abort", None)  # cooperative release of the abandoned session
            if ab is not None:
                try:
                    ab()
                except Exception:
                    pass
            return None, "timeout", None
        if "e" in box:
            return None, "error", box["e"]
        return box.get("r"), "ok", None

    def _drain_abandoned():
        """Wait a bounded interval for timed-out native workers before adapter teardown.

        Freeze the evaluation snapshot before calling this hook so cleanup cannot
        change the scored state or inference timing. Surviving native workers require
        a hard process exit to avoid unsafe interpreter finalization. Idempotent."""
        if _drained:
            return
        _drained.append(True)
        pending = [(t, d) for t, d in abandoned if t.is_alive()]
        if not pending:
            return
        log(
            f"[default] draining {len(pending)} abandoned generate worker(s), <= {ABANDONED_DRAIN_S:g}s"
        )
        limit = clock.time() + ABANDONED_DRAIN_S
        for _, d in pending:
            clock.wait(d, max(0.0, limit - clock.time()))
        live = sum(1 for t, _ in pending if t.is_alive())
        if live:
            log(
                f"[default] {live} abandoned worker(s) STILL inside a native call after "
                f"{ABANDONED_DRAIN_S:g}s — teardown continues; the runner hard-exits so interpreter "
                f"finalization cannot abort on them"
            )

    # Attach immediately so teardown can drain even if the loop raises.
    setattr(env, "_drain_abandoned", _drain_abandoned)

    def _run_step(step_fn, k):
        """One bounded per-move step -> a normalized generate-result dict (always has action/latency).
        A timeout or crash yields action=None and flags the adapter for a session rebuild."""
        t0 = clock.time()
        timeout, timeout_reason = _move_timeout()
        if getattr(env, "_invalid_run", None):
            return {
                "action": None,
                "latency": 0.0,
                "invalid_category": "environment",
                "invalid_reason": getattr(env, "_invalid_run")["reason"],
                "invalid_detail": getattr(env, "_invalid_run").get("detail"),
            }
        result, kind, err = _bounded_call(step_fn, timeout)
        if kind == "ok" and isinstance(result, dict):
            return result
        if kind == "timeout":
            log(f"[default] k={k} generation TIMED OUT (>{timeout}s, {timeout_reason}) — discarded")
            if timeout_reason == "provider_watchdog":
                return {
                    "action": None,
                    "latency": round(clock.time() - t0, 2),
                    "invalid_category": "provider",
                    "invalid_reason": "generation_watchdog_timeout",
                    "invalid_detail": f"generation exceeded {timeout:g}s",
                }
            return episode_records.timed_out_result(round(clock.time() - t0, 2), timeout_reason)
        log(f"[default] k={k} generation failed, recovering:\n{err}")
        return episode_records.generation_error(
            round(clock.time() - t0, 2), err or "adapter returned a non-object generation result"
        )

    def _charge(r, deliberation_s):
        # Charge the deliberation clock for one move and apply the per-move cap. A move whose
        # inference exceeds per_move_cap_s "times out" (like the paper's per-move compute cap): it
        # places NO action this turn and costs exactly the cap from the total clock. Returns the
        # action to actually execute (None if timed out) plus the timed_out flag.
        nonlocal spent
        if per_move_cap_s is not None and deliberation_s > per_move_cap_s:
            spent += per_move_cap_s
            return None, True
        spent += deliberation_s
        return r.get("action"), bool(r.get("timed_out"))

    def _commit(status, detail=""):
        fn = getattr(adapter, "commit_action", None)
        if fn is None:
            return
        try:
            fn(status, detail)
        except Exception:
            failure = traceback.format_exc()
            log(f"[default] commit_action({status}) failed:\n{failure}")
            _invalidate("provider", "action_commit_feedback_failed", failure)

    def _apply(k, t_k, r):
        """Measure inference -> advance the world -> land the delayed action if still valid."""
        nonlocal tau_prev
        terminal_after_action = False
        invalid = episode_records.result_invalid(r, adapter, clock_mode)
        if invalid is not None:
            _invalidate(invalid.category, invalid.reason, invalid.detail)
            return "invalid_run", False, False

        output_tokens = episode_records.frame_output_tokens(r)
        if clock_mode == "token" and tokens_per_frame is not None:
            if output_tokens is None:
                _invalidate(
                    "provider",
                    "missing_output_token_usage",
                    "finite token-clock decisions require provider or deterministic local usage",
                )
                return "invalid_run", False, False
            # API cells are timestamped from their complete response usage; local cells use the
            # runtime's complete emitted-token count. Protocol delimiters are part of those counts.
            action_output_tokens = output_tokens
            d_frames = action_output_tokens / tokens_per_frame
            d_ticks = d_frames * frame_ticks
            if r.get("post_horizon_output_tokens"):
                # The complete response lands past the horizon. The count charged, the allowance
                # rounded down to a whole token, can place it just short of the horizon.
                d_ticks = float(max_ticks) - float(t_k)
                d_frames = d_ticks / frame_ticks
            deliberation_s = d_ticks * delta_s
        elif clock_mode == "token":
            action_output_tokens = output_tokens
            # Unlimited B is true pause mode: generation adds no simulated delay, so the action
            # lands on the observed state. Observation cadence is still fixed because the
            # post-action phase below advances one complete frame before the next capture.
            d_frames = 0.0
            d_ticks = 0.0
            deliberation_s = 0.0
        else:
            action_output_tokens = None
            d_frames = None
            deliberation_s = max(0.0, r.get("latency", 0.0) or 0.0)
            d_ticks = deliberation_s / delta_s

        action, timed_out = _charge(r, deliberation_s)
        over_budget = think_budget_s is not None and spent >= think_budget_s
        if clock_mode == "token":
            tau_k = t_k + d_ticks
        else:
            position = getattr(env, "clock_position", None)
            tau_k = position() if callable(position) else env.tick_index()
            tau_k = max(float(tau_k), tau_prev)
        tau_prev = tau_k
        clock_target = min(tau_k, max_ticks)
        setattr(env, "_last_execution_result", None)
        setattr(env, "_last_coordinate_trace", None)

        def _land_action():
            nonlocal action, terminal_after_action
            if getattr(env, "_invalid_run", None):
                return "invalid_run"
            if action is None:
                return "skipped_timeout" if timed_out else "none"
            if over_budget:
                return "skipped_budget"
            if _past_deadline():
                return "not_applied_time_limit"
            if _episode_over():
                return "not_applied_terminal"
            if env.tick_index() >= max_ticks:
                return "not_applied_tick_limit"
            status = env.execute(action)
            if status == "failed":
                if getattr(env, "_last_rejected_by_guest", False):
                    # The guest refused a malformed action (HTTP 4xx, e.g. an unsupported key
                    # name). That is the model's action failing, not the environment.
                    return "rejected"
                _invalidate(
                    "environment", "action_execution_failed", getattr(env, "_last_exec_error", None)
                )
                return "invalid_run"
            # An applied action can decide the task immediately. Check while the environment is
            # still frozen at the action timestamp, before any action-to-observation remainder is
            # advanced. Keep the lifecycle status as "applied": the action did execute.
            terminal_after_action = _episode_over()
            return status

        # Token time has not advanced physically while the model generated, so advance it to the
        # token-derived action timestamp. Wall time already advanced while generation streamed: the
        # returned action must land immediately, without rounding up to (or waiting for) a frame.
        clock_before = env.tick_index()
        if clock_mode == "token":
            env.wait_until_tick(clock_target)
        clock_after = env.tick_index()
        _emit(
            TraceEvent.CLOCK_ADVANCED,
            {
                "decision": k,
                "phase": "inference_to_action",
                "clock_mode": clock_mode,
                "from_tick": clock_before,
                "requested_tick": clock_target,
                "to_tick": clock_after,
                "delta_ticks": clock_after - clock_before,
                "output_tokens": output_tokens,
                "tokens_per_frame": tokens_per_frame,
            },
        )
        # Timestamp the start of execution. Guest/CDP dispatch latency is an environment cost, not
        # extra model deliberation, so it must not move the committed action's recorded position.
        action_resolved_at = clock.time()
        status = _land_action()
        action_result = episode_records.action_resolved(
            decision=k,
            observation_id=r.get("observation_id"),
            action=action,
            status=status,
            terminal_after_action=terminal_after_action,
            action_clock_position=round(tau_k, 9),
            action_resolved_unix_s=action_resolved_at,
            environment_tick=env.tick_index(),
            execution_result=getattr(env, "_last_execution_result", None),
            coordinate_trace=getattr(env, "_last_coordinate_trace", None),
        )
        _emit(TraceEvent.ACTION_RESOLVED, action_result)
        if action is not None and status in {"applied", "wait", "terminal", "terminal_ignored"}:
            _emit(TraceEvent.ACTION_APPLIED, action_result)

        # Finishing early never increases observation frequency. After applying the action with the
        # environment frozen, advance only the remainder to the next scheduled frame boundary. A
        # generation spanning boundaries therefore drops those intermediate observations.
        next_observation_tick = env.tick_index()
        dropped_frames = 0
        post_action_rendered = False
        if (
            not terminal_after_action
            and status not in {"terminal", "not_applied_terminal", "invalid_run"}
            and env.tick_index() < max_ticks
        ):
            action_end_tick = max(float(tau_k), float(env.tick_index()))
            next_frame = max(_frame_at(t_k) + 1, _whole_up(action_end_tick / frame_ticks))
            next_observation_tick = min(_whole_up(next_frame * frame_ticks), max_ticks)
            post_before = env.tick_index()
            env.wait_until_tick(next_observation_tick)
            post_after = env.tick_index()
            # At an exact finite-token boundary, inference already advanced the world to the
            # observation tick before the action landed. The boundary wait above is consequently a
            # no-op. Commit a compositor frame at the same logical timestamp so the next capture
            # cannot reuse the pre-action surface/fence. Unlimited B normally advances a full frame
            # here and therefore does not need this zero-time barrier.
            if (
                clock_mode == "token"
                and action is not None
                and status == "applied"
                and abs(float(post_after) - float(post_before)) < 1e-9
            ):
                render_current_frame = getattr(env, "render_current_frame", None)
                if callable(render_current_frame):
                    post_action_rendered = render_current_frame() is not None
            crossed_boundaries = max(0, _frame_at(post_after) - _frame_at(t_k))
            dropped_frames = max(0, crossed_boundaries - 1)
            _emit(
                TraceEvent.CLOCK_ADVANCED,
                {
                    "decision": k,
                    "phase": "action_to_observation_boundary",
                    "clock_mode": clock_mode,
                    "from_tick": post_before,
                    "requested_tick": next_observation_tick,
                    "to_tick": post_after,
                    "delta_ticks": post_after - post_before,
                    "post_action_rendered": post_action_rendered,
                    "dropped_intermediate_frames": dropped_frames,
                },
            )

        if clock_mode == "token":
            setattr(
                env,
                "_dropped_intermediate_frames",
                int(getattr(env, "_dropped_intermediate_frames", 0) or 0) + dropped_frames,
            )

        observation_timestamp = r.get("observation_timestamp")
        frame_age_s = None
        if isinstance(observation_timestamp, (int, float)):
            frame_age_s = (
                max(0.0, (tau_k - r["observation_tick"]) * delta_s)
                if clock_mode == "token"
                else max(0.0, action_resolved_at - observation_timestamp)
            )
            ages = getattr(env, "frame_ages", None)
            if isinstance(ages, list):
                ages.append(frame_age_s)

        if status == "terminal":
            # Preserve the model's terminal text separately from the action for auditability.
            setattr(
                env,
                "_terminal_output",
                {
                    "thinking": r.get("thinking"),
                    "raw": r.get("raw"),
                    "action": action.action_type.value if action is not None else None,
                },
            )

        # A guest rejection carries the guest's reason (e.g. the unsupported key name) so the
        # model can correct the action instead of repeating it.
        _commit(status, getattr(env, "_last_exec_error", "") or "" if status == "rejected" else "")
        recorder = getattr(env, "record_lifecycle", None)
        if recorder is not None:
            try:
                recorder(
                    episode_records.lifecycle_record(
                        r,
                        decision=k,
                        clock_mode=clock_mode,
                        frame_age_s=frame_age_s,
                        action_clock_position=round(tau_k, 9),
                        action_resolved_unix_s=action_resolved_at,
                        frame_output_tokens=output_tokens,
                        action_output_tokens=action_output_tokens,
                        token_delay_frames=d_frames,
                        next_observation_tick=next_observation_tick,
                        post_action_rendered=post_action_rendered,
                        dropped_intermediate_frames=dropped_frames,
                        status=status,
                        terminal_after_action=terminal_after_action,
                        action=action,
                        execution_result=getattr(env, "_last_execution_result", None),
                        coordinate_trace=getattr(env, "_last_coordinate_trace", None),
                        clock_advance=(
                            getattr(getattr(env, "task", None), "_last_clock_advance", None)
                            if clock_mode == "token"
                            else None
                        ),
                        recorded_unix_s=clock.time(),
                    )
                )
            except Exception:
                failure = traceback.format_exc()
                log(f"[default] lifecycle audit write failed:\n{failure}")
                _invalidate("harness", "action_lifecycle_write_failed", failure)
        _checkpoint_trace()
        timeline.append((k, t_k, d_ticks, tau_k, action))
        log(
            f"[default] k={k} t_k={t_k} δ={d_ticks:g} τ={tau_k:g} act={_a(action)} -> {status}"
            + (f" tokens={output_tokens}" if clock_mode == "token" else "")
            + (" TIMEOUT" if timed_out else "")
        )
        return status, over_budget, terminal_after_action

    def _finalize():
        # Presentation boundary precedes scorer snapshot and input-release cleanup.
        setattr(env, "_active_end_unix_s", clock.time())
        # Freeze scorer input before cleanup: keyUp/mouseUp are real application-visible events and
        # must never alter the state scored at a terminal condition or at the horizon.
        setattr(env, "_think_spent", spent)
        snapshotter = getattr(getattr(env, "task", None), "evaluation_snapshot", None)
        setattr(env, "_termination_snapshot_captured", False)
        finalized_s = _world_s()
        snapshot_s = None
        if callable(snapshotter):
            try:
                with operation_budget(
                    "snapshot read", ENVIRONMENT_BUDGET.snapshot_read_s, clock=clock
                ):
                    snapshot = snapshotter()
                setattr(env, "_termination_snapshot", snapshot)
                snapshot_s = _world_s()
                setattr(env, "_termination_snapshot_captured", True)
                setattr(env, "_termination_snapshot_error", None)
            except Exception:
                failure = traceback.format_exc()
                setattr(env, "_termination_snapshot_error", failure)
                _invalidate("environment", "evaluation_snapshot_failed", failure)

        # Latch the outcome after snapshot validation and before best-effort input cleanup.
        if getattr(env, "_invalid_run", None):
            reason = "invalid_run"
        elif getattr(env, "_think_exhausted", False):
            reason = "think_budget"
        elif getattr(env, "_episode_over", False):
            reason = getattr(env, "_terminal_reason", None) or "game_decided"
        elif getattr(env, "_done", False):
            terminal_action = getattr(env, "_terminal_action", None)
            reason = "agent_fail" if terminal_action == ActionType.FAIL else "agent_done"
        elif _past_deadline():
            reason = "time_limit"
        elif env.tick_index() >= max_ticks:
            reason = "tick_limit"
        else:
            reason = "completed"
        setattr(env, "_stop_reason", reason)
        setattr(env, "_termination", _termination(reason, finalized_s, snapshot_s))
        release_inputs = getattr(env, "release_inputs", None)
        if callable(release_inputs):
            try:
                release_inputs()
            except Exception:
                log("[cleanup] failed to release stateful inputs:\n" + traceback.format_exc())

    def _termination(reason, finalized_s, snapshot_s):
        """What the report reads about the end: the stop reason, the end time its rule gives
        and when the read of the scored snapshot returned (None without one), in world seconds.
        An end the loop first sees while finalizing ends at ``finalized_s``."""
        if reason in {"time_limit", "tick_limit"}:
            end_s = max_ticks * delta_s
        elif reason in {"invalid_run", "think_budget"}:
            end_s = end_signals.get(reason, finalized_s)
        elif getattr(env, "_episode_over", False):  # the task's own reason
            end_s = end_signals.get("environment", finalized_s)
        else:  # agent_done, agent_fail and completed have no end signal
            end_s = finalized_s
        return {
            "stop_reason": reason,
            "end_s": round(end_s, 9),
            "snapshot_s": None if snapshot_s is None else round(snapshot_s, 9),
            "snapshot_delay_s": None if snapshot_s is None else round(snapshot_s - end_s, 9),
        }

    def _mark_budget_exhausted():
        end_signals.setdefault("think_budget", _world_s())
        setattr(env, "_think_exhausted", True)
        log(
            f"[think] deliberation budget exhausted: {spent:.1f}s >= {think_budget_s:g}s -> forfeit"
        )

    def _environment_done():
        queried_at = _world_s()
        return _end_reported(env.done(), queried_at)

    k = 0

    def _decide():
        """Run one decision; False once the episode is over."""
        nonlocal k
        if getattr(env, "_invalid_run", None):
            return False
        try:
            obs = env.capture()
        except Exception:
            _invalidate("environment", "observation_capture_failed", traceback.format_exc())
            return False
        if not _accept_observation(obs):
            return False
        _emit(TraceEvent.OBSERVATION_CAPTURED, episode_records.observation_captured(obs, k))
        t_k = obs.tick_number
        metadata = getattr(obs, "metadata", {}) or {}

        remaining_clock_tokens = None
        if clock_mode == "token" and tokens_per_frame is not None:
            allowance = (float(max_ticks) - float(t_k)) / frame_ticks * tokens_per_frame
            # Float noise grows with the allowance; at a large B it is past a fixed 1e-9.
            remaining_clock_tokens = max(0, math.floor(allowance + max(1e-9, allowance * 1e-12)))
            if remaining_clock_tokens <= 0:
                env.wait_until_tick(max_ticks)
                return False
        native_output_limit, budget = episode_records.decision_budget(
            adapter, remaining_clock_tokens, clock_mode
        )
        adapter.configure_decision(budget)

        def _step(obs=obs, k=k):
            adapter.stream_prefill(obs)
            return adapter.stream_generate(tick=k)

        usage_before = episode_records.usage_snapshot(adapter)
        _emit(
            TraceEvent.INFERENCE_STARTED,
            episode_records.inference_started(obs, k, usage_before, native_output_limit, budget),
        )
        r = _run_step(_step, k)
        episode_records.annotate_result(
            r,
            decision=k,
            observation=obs,
            native_output_limit=native_output_limit,
            budget=budget,
            tokens_per_frame=tokens_per_frame,
        )
        usage_after = episode_records.usage_snapshot(adapter)
        if usage_before is not None and usage_after is not None:
            r["frame_usage"] = episode_records.frame_usage(usage_before, usage_after)
        # The runner kept the episode's settlements on the environment; it reports every decision.
        if not hasattr(env, "_settlements"):
            env._settlements = []
        env._settlements.append(episode_records.account_tokens(r, remaining_clock_tokens))
        observation_id = metadata.get("observation_id")
        for chunk in episode_records.output_chunks(r, k, observation_id):
            _emit(TraceEvent.OUTPUT_CHUNK_RECEIVED, chunk)
        _emit(
            TraceEvent.INFERENCE_COMPLETED,
            episode_records.inference_completed(r, k, observation_id),
        )
        episode_records.audit_output_caps(adapter, r, budget)
        failure = episode_records.finalize_decision_audit(adapter, r)
        if failure is not None:
            log(f"[default] trajectory audit write failed:\n{failure}")
            _invalidate("harness", "trajectory_write_failed", failure)
        action_protocol = r.get("action_protocol") or {}
        proposed = r.get("action")
        if proposed is not None and proposed.action_type in {ActionType.DONE, ActionType.FAIL}:
            r["action"] = Action.wait()
            action_protocol = episode_records.terminal_wait_protocol(action_protocol)
            r["action_protocol"] = action_protocol
        if r.get("action") is not None and not action_protocol.get("protocol_failure"):
            _emit(
                TraceEvent.ACTION_PROPOSED,
                episode_records.action_proposed(r, k, observation_id),
            )
        status, over_budget, terminal_after_action = _apply(k, t_k, r)
        k += 1
        if status == "invalid_run":
            return False
        if over_budget:
            _mark_budget_exhausted()
            return False
        if status in ("terminal", "not_applied_terminal"):
            return False
        if terminal_after_action:
            return False
        if _past_deadline():
            return False
        return True

    # An exception that escapes the loop still ends the episode through _finalize, which freezes
    # the snapshot and latches the stop reason; the run is invalid.
    try:
        while not _environment_done() and env.tick_index() < max_ticks:
            if not _decide():
                break
    except Exception:
        failure = traceback.format_exc()
        log(f"[default] decision loop raised:\n{failure}")
        # The whole traceback, for the summary's error; the invalid-run detail keeps its end.
        setattr(env, "_loop_error", failure)
        _invalidate("runner", "run_exception", failure)
    _finalize()
    return timeline


def _a(action):
    return None if action is None else (action.action_type.value, action.parameters)
