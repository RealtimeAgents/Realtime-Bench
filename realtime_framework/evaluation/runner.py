from __future__ import annotations

# Loop
"""Frame-driven decision loop for the wall and token clocks.

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
import hashlib as hashlib, math as math, threading as threading, time as time, traceback as traceback

from realtime_framework.evaluation.config import DEFAULT_MODE

from realtime_framework.entity.actions import Action
from realtime_framework.entity.actions import ActionType
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.evaluation.clock import token_record
from realtime_framework.sandbox.observation.som import valid_inspection_proof

RUNNER_TIMING = RunnerTimingConfig.from_environ()  # read once, at import
# Small settle margin added to the wall-clock bound so a well-behaved call that is about to return
# at the deadline can finish; per-move compute caps receive no grace period.
DEADLINE_GRACE_S = RUNNER_TIMING.deadline_grace_s
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

    def __init__(self, threshold=OBSERVATION_FREEZE_N, clock=time.monotonic):
        self.threshold = max(2, int(threshold))
        self.clock = clock
        self._fingerprint = None
        self._first_seen_at = None
        self._count = 0

    @staticmethod
    def fingerprint(observation):
        metadata = getattr(observation, "metadata", {}) or {}
        digest = metadata.get("screenshot_sha256")
        if not digest:
            shot = getattr(observation, "screenshot", b"") or b""
            digest = hashlib.sha256(shot).hexdigest()
        return (
            metadata.get("observation_id"),
            getattr(observation, "tick_number", None),
            digest,
        )

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


def run_stream(
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
):
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
    observation_watchdog = ObservationStreamWatchdog()
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
                        "detail": failure[:2000],
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
                value["detail"] = str(detail)[:2000]
            setattr(env, "_invalid_run", value)
            log(f"[invalid-run] {category}:{reason}" + (f" — {detail}" if detail else ""))
            _emit(TraceEvent.RUN_INVALIDATED, value)
        return getattr(env, "_invalid_run")

    def _action_record(action):
        if action is None:
            return None
        return {
            "action_type": action.action_type.value,
            "parameters": action.parameters,
            "coordinate_space": action.coordinate_space,
            "coordinate_transform": action.coordinate_transform,
        }

    def _usage_snapshot():
        try:
            value = getattr(adapter, "token_usage", None)
            value = value() if callable(value) else value
            return dict(value) if isinstance(value, dict) else None
        except Exception:
            return None

    def _usage_number(usage, *names):
        if not isinstance(usage, dict):
            return None
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                return int(value)
        return None

    def _clock_output_tokens(r):
        """Return the complete native output count and its clock source.

        Remote cells must use the API's response-level output usage. Local native runtimes use the
        token IDs emitted by that runtime.
        """
        provider_missing = r.get("provider_usage_missing_reason")
        if not provider_missing:
            value = r.get("provider_output_tokens")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                return int(value), "provider_reported_output_usage"
            value = _usage_number(r.get("usage"), "output_tokens", "completion_tokens")
            if value is not None:
                return value, "provider_reported_output_usage"
            value = _usage_number(r.get("frame_usage"), "output_tokens", "completion_tokens")
            if value is not None:
                return value, "provider_reported_output_usage_delta"
        value = r.get("native_received_output_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value), (
                r.get("native_received_output_tokens_source") or "native_runtime_output_tokens"
            )
        return None, None

    def _frame_output_tokens(r):
        value = r.get("clock_output_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value)
        return _clock_output_tokens(r)[0]

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
        done = threading.Event()

        def worker():
            try:
                box["r"] = fn()
            except Exception:
                box["e"] = traceback.format_exc()
            finally:
                done.set()

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        if not done.wait(timeout):
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
        limit = time.time() + ABANDONED_DRAIN_S
        for _, d in pending:
            d.wait(max(0.0, limit - time.time()))
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
        t0 = time.time()
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
                    "latency": round(time.time() - t0, 2),
                    "invalid_category": "provider",
                    "invalid_reason": "generation_watchdog_timeout",
                    "invalid_detail": f"generation exceeded {timeout:g}s",
                }
            return {
                "action": None,
                "latency": round(time.time() - t0, 2),
                "timed_out": True,
                "harness_timeout_reason": timeout_reason,
                "provider_usage_missing_reason": "generation_aborted_at_wall_timeout",
            }
        log(f"[default] k={k} generation failed, recovering:\n{err}")
        return {
            "action": None,
            "latency": round(time.time() - t0, 2),
            "error": True,
            "invalid_category": "provider",
            "invalid_reason": "generation_error",
            "invalid_detail": err or "adapter returned a non-object generation result",
        }

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
        proof = r.get("inspection") if isinstance(r, dict) else None
        inspection_required = getattr(adapter, "tool_runtime", None) is not None
        inspection_ok = not inspection_required or valid_inspection_proof(
            proof,
            r.get("observation_id"),
        )
        if r.get("invalid_reason"):
            _invalidate(
                r.get("invalid_category") or "provider",
                r["invalid_reason"],
                r.get("invalid_detail"),
            )
            return "invalid_run", False, False
        # A harness-aborted wall-clock call has no returned model decision to inspect. Keep
        # charging its elapsed time and recording unknown usage, but never mistake this synthetic
        # timeout for a model response missing its observation-bound inspection proof.
        harness_wall_timeout = (
            clock_mode == "wall"
            and r.get("timed_out")
            and r.get("action") is None
            and r.get("harness_timeout_reason") in {"wall_deadline", "per_move_cap"}
        )
        if not inspection_ok and not harness_wall_timeout:
            _invalidate(
                "tool",
                "inspection_protocol_error",
                (proof or {}).get("protocol_error") or (proof or {}).get("status"),
            )
            return "invalid_run", False, False

        output_tokens = _frame_output_tokens(r)
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
            d_ticks = action_output_tokens / tokens_per_frame * frame_ticks
            deliberation_s = d_ticks * delta_s
        elif clock_mode == "token":
            action_output_tokens = output_tokens
            # Unlimited B is true pause mode: generation adds no simulated delay, so the action
            # lands on the observed state. Observation cadence is still fixed because the
            # post-action phase below advances one complete frame before the next capture.
            d_ticks = 0.0
            deliberation_s = 0.0
        else:
            action_output_tokens = None
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
        action_resolved_at = time.time()
        status = _land_action()
        action_result = {
            "decision": k,
            "observation_id": r.get("observation_id"),
            "action": _action_record(action),
            "status": status,
            "terminal_after_action": terminal_after_action,
            "action_clock_position": round(tau_k, 9),
            "action_resolved_unix_s": action_resolved_at,
            "environment_tick": env.tick_index(),
            "execution_result": getattr(env, "_last_execution_result", None),
            "coordinate_trace": getattr(env, "_last_coordinate_trace", None),
        }
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
                max(0.0, (tau_k - r.get("observation_tick", t_k)) * delta_s)
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
                    {
                        "decision": k,
                        "generation_id": (r.get("realtime_runtime") or {}).get("generation_id"),
                        "observation_id": r.get("observation_id"),
                        "observation_batch": r.get("observation_batch"),
                        "observation_tick": r.get("observation_tick", t_k),
                        "observation_timestamp": observation_timestamp,
                        "frame_age_at_action_s": round(frame_age_s, 6)
                        if frame_age_s is not None
                        else None,
                        "frame_age_clock": (
                            "simulated_token_clock" if clock_mode == "token" else "wall_clock"
                        ),
                        "frame_age_definition": (
                            "action_simulated_time_minus_observation_simulated_time"
                            if clock_mode == "token"
                            else "action_resolution_wall_time_minus_observation_wall_time"
                        ),
                        # Compatibility field: this is the complete adapter decision duration, not
                        # pure provider compute and not time to first token.
                        "model_latency_s": r.get("latency"),
                        "harness_timeout_reason": r.get("harness_timeout_reason"),
                        "model_latency_scope": (
                            "observation_preparation_through_complete_response_processing"
                        ),
                        "decision_wall_latency_s": r.get("decision_wall_latency_s"),
                        "provider_call_wall_latency_s": r.get("provider_call_wall_latency_s"),
                        "runtime_turn_wall_latency_s": r.get("runtime_turn_wall_latency_s"),
                        "provider_ttft_s": r.get("provider_ttft_s"),
                        "provider_ttft_source": r.get("provider_ttft_source"),
                        "provider_ttft_missing_reason": r.get("provider_ttft_missing_reason"),
                        "latency_clock": "wall_monotonic",
                        "action_clock_position": round(tau_k, 9),
                        "action_resolved_unix_s": action_resolved_at,
                        "frame_output_tokens": output_tokens,
                        "token_accounting": r.get("token_accounting"),
                        "action_output_tokens": action_output_tokens,
                        "provider_output_tokens": r.get("provider_output_tokens"),
                        "provider_reasoning_output_tokens": r.get(
                            "provider_reasoning_output_tokens"
                        ),
                        "provider_visible_output_tokens": r.get("provider_visible_output_tokens"),
                        "provider_audio_output_tokens": r.get("provider_audio_output_tokens"),
                        "provider_token_normalization_schema": r.get(
                            "provider_token_normalization_schema"
                        ),
                        "provider_token_semantics": r.get("provider_token_semantics"),
                        "clock_output_tokens": r.get("clock_output_tokens"),
                        "clock_output_tokens_source": r.get("clock_output_tokens_source"),
                        "remaining_clock_tokens": r.get("remaining_clock_tokens"),
                        "unclamped_clock_output_tokens": r.get("unclamped_clock_output_tokens"),
                        "post_horizon_output_tokens": r.get("post_horizon_output_tokens", 0),
                        "provider_response_status": r.get("provider_response_status"),
                        "provider_response_status_reason": r.get("provider_response_status_reason"),
                        "action_protocol": r.get("action_protocol"),
                        "token_accounting_source": r.get("clock_output_tokens_source"),
                        "token_delay_frames": d_ticks if clock_mode == "token" else None,
                        "next_observation_tick": next_observation_tick,
                        "post_action_rendered": post_action_rendered,
                        "dropped_intermediate_frames": dropped_frames,
                        "decision_output_cap": r.get("decision_output_cap"),
                        "native_output_limit": r.get("native_output_limit"),
                        "runtime_output_token_cap": r.get("runtime_output_token_cap"),
                        "provider_output_token_cap": r.get("provider_output_token_cap"),
                        "transport_output_token_cap": r.get("transport_output_token_cap"),
                        "handoff_token_cap": r.get("handoff_token_cap"),
                        "action_token_cap": r.get("action_token_cap"),
                        "status": status,
                        "terminal_after_action": terminal_after_action,
                        "action": _action_record(action),
                        "execution_result": getattr(env, "_last_execution_result", None),
                        "coordinate_trace": getattr(env, "_last_coordinate_trace", None),
                        "clock_advance": (
                            getattr(getattr(env, "task", None), "_last_clock_advance", None)
                            if clock_mode == "token"
                            else None
                        ),
                        "recorded_unix_s": time.time(),
                    }
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
        setattr(env, "_active_end_unix_s", time.time())
        # Freeze scorer input before cleanup: keyUp/mouseUp are real application-visible events and
        # must never alter the state scored at a terminal condition or at the horizon.
        setattr(env, "_think_spent", spent)
        snapshotter = getattr(getattr(env, "task", None), "evaluation_snapshot", None)
        setattr(env, "_termination_snapshot_captured", False)
        finalized_s = _world_s()
        snapshot_s = None
        if callable(snapshotter):
            try:
                setattr(env, "_termination_snapshot", snapshotter())
                setattr(env, "_termination_snapshot_captured", True)
                setattr(env, "_termination_snapshot_error", None)
                snapshot_s = finalized_s
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
        and when the scored snapshot was frozen (None without one), in world seconds. An end the
        loop first sees while finalizing ends at ``finalized_s``."""
        if reason in {"time_limit", "tick_limit"}:
            end_s = max_ticks * delta_s
        else:
            signal = reason if reason in {"invalid_run", "think_budget"} else "environment"
            end_s = end_signals.get(signal, finalized_s)
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

    def _record_observation(obs, decision=None):
        metadata = getattr(obs, "metadata", {}) or {}
        _emit(
            TraceEvent.OBSERVATION_CAPTURED,
            {
                "decision": decision,
                "observation_id": metadata.get("observation_id"),
                "tick": obs.tick_number,
                "timestamp": obs.timestamp,
                "screenshot_sha256": metadata.get("screenshot_sha256")
                or hashlib.sha256(obs.screenshot or b"").hexdigest(),
                "capture_latency_s": metadata.get("capture_latency_s"),
                "interactables_status": (metadata.get("interactables") or {})
                .get("scene", {})
                .get("status"),
            },
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
        _record_observation(obs, decision=k)
        t_k = obs.tick_number
        metadata = getattr(obs, "metadata", {}) or {}
        observation_batch = [
            {
                "observation_id": metadata.get("observation_id"),
                "tick": obs.tick_number,
                "timestamp": obs.timestamp,
            }
        ]

        native_output_limit = getattr(
            getattr(adapter, "config", None), "native_output_limit", None
        )
        remaining_clock_tokens = None
        if clock_mode == "token" and tokens_per_frame is not None:
            remaining_clock_tokens = max(
                0,
                math.floor(
                    (float(max_ticks) - float(t_k)) / frame_ticks * tokens_per_frame + 1e-9
                ),
            )
            if remaining_clock_tokens <= 0:
                env.wait_until_tick(max_ticks)
                return False
        # None when the model declares no native limit and no token clock bounds the episode.
        runtime_output_cap = (
            native_output_limit
            if remaining_clock_tokens is None
            else remaining_clock_tokens
            if native_output_limit is None
            else min(native_output_limit, remaining_clock_tokens)
        )
        # Bound transports that support a per-turn cap by the episode's remaining charged-token
        # capacity. Drivers without an exact per-turn cap may still return terminal usage beyond T;
        # that excess is retained for billing audit but excluded from the episode clock below.
        if remaining_clock_tokens is not None:
            provider_output_token_cap = runtime_output_cap
        else:
            unbounded_output = bool(getattr(adapter, "supports_unbounded_output", False))
            provider_output_token_cap = None if unbounded_output else runtime_output_cap
        transport_cap_mapper = getattr(adapter, "transport_output_token_cap", None)
        transport_output_cap = (
            transport_cap_mapper(provider_output_token_cap)
            if callable(transport_cap_mapper)
            else provider_output_token_cap
        )
        if not (
            transport_output_cap is None
            or transport_output_cap == "inf"
            or (
                isinstance(transport_output_cap, int)
                and not isinstance(transport_output_cap, bool)
                and transport_output_cap > 0
            )
        ):
            raise ValueError(
                "adapter transport output cap must be None, a positive integer, or 'inf'"
            )
        adapter.configure_decision(
            DecisionBudget(
                runtime_output_tokens=runtime_output_cap,
                provider_output_tokens=provider_output_token_cap,
                transport_output_tokens=transport_output_cap,
                clock_mode=clock_mode,
            )
        )

        def _step(obs=obs, k=k):
            adapter.stream_prefill(obs)
            return adapter.stream_generate(tick=k)

        usage_before = _usage_snapshot()
        _emit(
            TraceEvent.INFERENCE_STARTED,
            {
                "decision": k,
                "observation_id": metadata.get("observation_id"),
                "observation_tick": getattr(obs, "tick_number", t_k),
                "observation_batch": observation_batch,
                "usage_before": usage_before,
                "runtime_output_token_cap": runtime_output_cap,
                "provider_output_token_cap": provider_output_token_cap,
                "transport_output_token_cap": transport_output_cap,
                "native_output_limit": native_output_limit,
            },
        )
        r = _run_step(_step, k)
        r.update(
            {
                "decision": k,
                "observation_id": metadata.get("observation_id"),
                "observation_tick": getattr(obs, "tick_number", t_k),
                "observation_batch": observation_batch,
                "observation_timestamp": getattr(obs, "timestamp", None),
                "decision_output_cap": runtime_output_cap,
                "native_output_limit": native_output_limit,
                "runtime_output_token_cap": runtime_output_cap,
                "provider_output_token_cap": r.get(
                    "provider_output_token_cap", provider_output_token_cap
                ),
                "transport_output_token_cap": r.get(
                    "transport_output_token_cap", transport_output_cap
                ),
                "remaining_clock_tokens": remaining_clock_tokens,
                "clock_delay_clock": (
                    "simulated_token_clock" if clock_mode == "token" else "wall_clock"
                ),
                "clock_delay_definition": (
                    "complete_clock_output_tokens_divided_by_tokens_per_frame"
                    if clock_mode == "token" and tokens_per_frame is not None
                    else "paused_environment_no_generation_delay"
                    if clock_mode == "token"
                    else "complete_decision_wall_latency"
                ),
            }
        )
        runtime_metadata = r.get("realtime_runtime")
        if isinstance(runtime_metadata, dict):
            for name in ("provider_response_status", "provider_response_status_reason"):
                if r.get(name) is None and runtime_metadata.get(name) is not None:
                    r[name] = runtime_metadata[name]
        usage_after = _usage_snapshot()
        if usage_before is not None and usage_after is not None:
            keys = set(usage_before) | set(usage_after)
            r["frame_usage"] = {
                name: max(
                    0, int(usage_after.get(name, 0) or 0) - int(usage_before.get(name, 0) or 0)
                )
                for name in keys
                if isinstance(usage_before.get(name, 0), (int, float))
                and isinstance(usage_after.get(name, 0), (int, float))
            }
        clock_output_tokens, clock_output_tokens_source = _clock_output_tokens(r)
        if (
            remaining_clock_tokens is not None
            and clock_output_tokens is not None
            and clock_output_tokens > remaining_clock_tokens
        ):
            r["unclamped_clock_output_tokens"] = clock_output_tokens
            r["post_horizon_output_tokens"] = clock_output_tokens - remaining_clock_tokens
            clock_output_tokens = remaining_clock_tokens
            clock_output_tokens_source = (
                f"episode_horizon_clamped:{clock_output_tokens_source or 'unknown'}"
            )
        r["clock_output_tokens"] = clock_output_tokens
        r["clock_output_tokens_source"] = clock_output_tokens_source
        r["action_output_tokens"] = clock_output_tokens
        r["remaining_clock_tokens"] = remaining_clock_tokens
        r.setdefault("unclamped_clock_output_tokens", None)
        r.setdefault("post_horizon_output_tokens", 0)
        r["token_accounting"] = token_record(r)
        for name in (
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_total_tokens",
            "provider_reasoning_output_tokens",
            "provider_visible_output_tokens",
            "provider_audio_output_tokens",
            "provider_output_tokens_source",
            "provider_reasoning_output_tokens_source",
            "provider_visible_output_tokens_source",
            "provider_visible_output_tokens_missing_reason",
            "provider_audio_output_tokens_source",
            "provider_audio_output_tokens_missing_reason",
            "provider_token_normalization_schema",
            "provider_token_semantics",
            "provider_protocol_family",
            "provider_upstream",
        ):
            r[name] = r["token_accounting"].get(name)
        accounting = getattr(env, "_token_accounting", None)
        if accounting is None:
            accounting = TokenAccounting()
            env._token_accounting = accounting
        accounting.add(r["token_accounting"])
        chunks = r.get("output_chunks")
        if not isinstance(chunks, list):
            chunks = (
                [{"content": r.get("raw"), "final": True}]
                if r.get("raw") not in {None, ""}
                else []
            )
        for chunk_index, chunk in enumerate(chunks):
            chunk_record = chunk if isinstance(chunk, dict) else {"content": str(chunk)}
            _emit(
                TraceEvent.OUTPUT_CHUNK_RECEIVED,
                {
                    "decision": k,
                    "observation_id": metadata.get("observation_id"),
                    "chunk_index": chunk_index,
                    "final": bool(chunk_record.get("final", chunk_index == len(chunks) - 1)),
                    "content": chunk_record.get("content", ""),
                    "channel": chunk_record.get("channel"),
                    "output_tokens": chunk_record.get("output_tokens"),
                    "native_token_count": chunk_record.get("native_token_count"),
                    "function_call": chunk_record.get("function_call"),
                    "transport_event": chunk_record.get("transport_event"),
                    # The trace event itself is appended after inference completes. These fields are
                    # captured synchronously when the collector receives the provider event.
                    "received_wall_time_ns": chunk_record.get("received_wall_time_ns"),
                    "received_monotonic_ns": chunk_record.get("received_monotonic_ns"),
                    "receive_timestamp_source": chunk_record.get(
                        "receive_timestamp_source", "unavailable"
                    ),
                    "trace_recorded_after_inference": True,
                },
            )
        _emit(
            TraceEvent.INFERENCE_COMPLETED,
            {
                "decision": k,
                "observation_id": metadata.get("observation_id"),
                "latency_s": r.get("latency"),
                "latency_compatibility_scope": (
                    "observation_preparation_through_complete_response_processing"
                ),
                "decision_wall_latency_s": r.get("decision_wall_latency_s"),
                "provider_call_wall_latency_s": r.get("provider_call_wall_latency_s"),
                "runtime_turn_wall_latency_s": r.get("runtime_turn_wall_latency_s"),
                "provider_ttft_s": r.get("provider_ttft_s"),
                "provider_ttft_source": r.get("provider_ttft_source"),
                "provider_ttft_missing_reason": r.get("provider_ttft_missing_reason"),
                "latency_clock": "wall_monotonic",
                "usage": r.get("usage"),
                "frame_usage": r.get("frame_usage"),
                "token_accounting": r.get("token_accounting"),
                "provider_output_tokens": r.get("provider_output_tokens"),
                "provider_reasoning_output_tokens": r.get("provider_reasoning_output_tokens"),
                "provider_visible_output_tokens": r.get("provider_visible_output_tokens"),
                "provider_audio_output_tokens": r.get("provider_audio_output_tokens"),
                "provider_token_normalization_schema": r.get(
                    "provider_token_normalization_schema"
                ),
                "provider_token_semantics": r.get("provider_token_semantics"),
                "clock_output_tokens": r.get("clock_output_tokens"),
                "clock_output_tokens_source": r.get("clock_output_tokens_source"),
                "remaining_clock_tokens": r.get("remaining_clock_tokens"),
                "unclamped_clock_output_tokens": r.get("unclamped_clock_output_tokens"),
                "post_horizon_output_tokens": r.get("post_horizon_output_tokens", 0),
                "provider_output_token_cap": r.get("provider_output_token_cap"),
                "transport_output_token_cap": r.get("transport_output_token_cap"),
                "provider_response_status": r.get("provider_response_status"),
                "provider_response_status_reason": r.get("provider_response_status_reason"),
                "stale": bool(r.get("stale")),
                "timed_out": bool(r.get("timed_out")),
                "error": bool(r.get("error")),
                "invalid_reason": r.get("invalid_reason"),
            },
        )
        handoff_cap = int(getattr(adapter, "_frame_handoff_max_tokens", 0) or 0)
        action_cap = int(
            getattr(
                adapter,
                "_frame_action_max_tokens",
                runtime_output_cap,
            )
            or 0
        )
        # A decision that was invalidated before generation (e.g. inspection_protocol_error)
        # never recomputed the adapter's per-decision caps, so they still describe the previous
        # decision; auditing them against this decision's allowance would raise spuriously.
        if (
            provider_output_token_cap is not None
            and r.get("invalid_reason") is None
            and handoff_cap + action_cap > provider_output_token_cap
        ):
            raise RuntimeError(
                f"adapter decision output caps exceed provider transport allowance: "
                f"handoff={handoff_cap} + action={action_cap} > {provider_output_token_cap}"
            )
        r.update(
            {
                "handoff_token_cap": handoff_cap,
                "action_token_cap": action_cap,
                "unclamped_clock_output_tokens": r.get("unclamped_clock_output_tokens"),
                "post_horizon_output_tokens": r.get("post_horizon_output_tokens", 0),
            }
        )
        finalize_decision_audit = getattr(adapter, "finalize_decision_audit", None)
        if callable(finalize_decision_audit):
            try:
                finalize_decision_audit(r)
            except Exception:
                failure = traceback.format_exc()
                log(f"[default] trajectory audit write failed:\n{failure}")
                _invalidate("harness", "trajectory_write_failed", failure)
                r.update(
                    {
                        "invalid_category": "harness",
                        "invalid_reason": "trajectory_write_failed",
                        "invalid_detail": failure,
                    }
                )
        action_protocol = r.get("action_protocol") or {}
        proposed = r.get("action")
        if proposed is not None and proposed.action_type in {ActionType.DONE, ActionType.FAIL}:
            r["action"] = Action.wait()
            action_protocol = {
                **action_protocol,
                "source": "invalid_action_fallback_wait",
                "valid_action": False,
                "protocol_failure": "unsupported_terminal_action",
            }
            r["action_protocol"] = action_protocol
        if r.get("action") is not None and not action_protocol.get("protocol_failure"):
            _emit(
                TraceEvent.ACTION_PROPOSED,
                {
                    "decision": k,
                    "observation_id": metadata.get("observation_id"),
                    "action": _action_record(r.get("action")),
                    "thinking": r.get("thinking"),
                    "protocol": action_protocol,
                    "action_output_tokens": r.get("action_output_tokens"),
                },
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
        _invalidate("runner", "run_exception", failure)
    _finalize()
    return timeline


def _a(action):
    return None if action is None else (action.action_type.value, action.parameters)


# Episode runner
"""End-to-end runner for persistent-session adapters on the real desktop container.

Token-clock decisions are charged by output-token consumption and wall-clock decisions by latency;
the world advances by that delay before the proposed action can land. Desktop lifecycle and task
scoring use their shared owners.

Usage: python -m realtime_framework.evaluation.runner --model gpt-realtime-2 --task <task-id>
       [--fps 1] [--max-minutes 5] --run-id NAME --results-dir DIR
"""
import argparse as argparse
import json as json
import logging as logging
import os as os
import sys as sys
import time as time
from datetime import datetime as datetime
from pathlib import Path as Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realtime_framework.sandbox.desktop as R
import realtime_framework.providers.registry as catalog
from realtime_framework.harnesses.registry import configure_agent
from realtime_framework.harnesses.registry import create_agent
from realtime_framework.entity.config import DEFAULT_TOKENS_PER_FRAME
from realtime_framework.entity.config import EnvironmentConfig
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import phase1_context_policy
from realtime_framework.providers.preflight import UnsupportedTokenBudget
from realtime_framework.providers.preflight import check_episode
from realtime_framework.providers.preflight import check_token_budget
from realtime_framework.sandbox.observation.capture import BrowserSceneCapture
from realtime_framework.sandbox.guest import GuestClient
from realtime_framework.sandbox.actions import GuestEnvironment
from realtime_framework.evaluation.reporting import capture_generation_artifact
from realtime_framework.evaluation.reporting import resolve_run_identity
from realtime_framework.evaluation.reporting import save_evaluation
from realtime_framework.evaluation.reporting import summarize_cost
from realtime_framework.evaluation.config import OUTPUT_CLOCK_POLICY
from realtime_framework.evaluation.config import INTERACTABLES_PROTOCOL
from realtime_framework.evaluation.config import INTERACTABLES_REPRESENTATION
from realtime_framework.evaluation.config import INTERACTABLES_SCHEMA
from realtime_framework.evaluation.config import config_fingerprint
from realtime_framework.evaluation.config import effective_run_config
from realtime_framework.evaluation.config import harness_fingerprint, harness_version
from realtime_framework.evaluation.reporting import summarize_run_metrics
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.evaluation.reporting import episode_outcome
from realtime_framework.evaluation.agent_trace import AgentTraceSink
from realtime_framework.evaluation.agent_trace import close_agent_trace
from realtime_framework.entity.records import canonical_sha256
import realtime_tasks.registry as custom_tasks
from realtime_framework.sandbox.observation.som import ToolRuntime


def _model_identity(requested_model, requested_adapter=None):
    """Separate the stable catalog key from the actual backend model and adapter."""
    if catalog.has(requested_model):
        entry = catalog.get(requested_model)
        return {
            "adapter": entry.adapter_type,
            "model": requested_model,
            "backend_model": entry.model_name,
        }
    return {
        "adapter": requested_adapter,
        "model": requested_model,
        "backend_model": requested_model,
    }


def _metric_avg(env, name):
    values = list(getattr(env, name, []) or []) if env is not None else []
    return round(sum(values) / len(values), 6) if values else None


def _metric_max(env, name):
    values = list(getattr(env, name, []) or []) if env is not None else []
    return round(max(values), 6) if values else None


def _effective_capture_fps(env):
    values = list(getattr(env, "capture_times", []) or []) if env is not None else []
    if len(values) < 2 or values[-1] <= values[0]:
        return None
    return round((len(values) - 1) / (values[-1] - values[0]), 6)


def _atomic_write_json(path, obj):
    """Write and fsync a sibling temporary file, then atomically replace the target.

    Interrupted writes preserve the previous complete result for resume checks."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class EnvironmentInvalidError(RuntimeError):
    """Startup health checks failed; the result must not be attributed to the model."""


def _invalid_evaluation(result):
    """Return a structured reason when an evaluator response is not safe to score."""
    if not isinstance(result, dict):
        return "evaluation_not_an_object"
    if result.get("error"):
        return "evaluation_error"
    if result.get("evaluation_valid") is False:
        return "evaluation_declared_invalid"
    if result.get("ok") is False:
        return "evaluation_backend_unhealthy"
    if result.get("invalid_reasons"):
        return "evaluation_invalid_reasons"
    if result.get("baseline_suspect") is True:
        return "evaluation_baseline_suspect"
    if type(result.get("resolved")) is not bool:
        return "evaluation_missing_boolean_resolved"
    if type(result.get("evaluation_version")) is not int:
        return "evaluation_missing_version"
    return None


def _invalid_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return "snapshot_not_an_object"
    state = snapshot.get("state")
    if not isinstance(state, dict) or not state:
        return "snapshot_missing_environment_state"
    if state.get("ok") is False:
        return "snapshot_environment_unhealthy"
    evaluation = snapshot.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("error"):
        return "snapshot_evaluator_error"
    return None


# Capture, simulated-time advancement, and action execution live in realtime_framework.sandbox.actions.
from realtime_framework.entity.config import task_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        default=None,
        help="adapter_type; optional when --model is a catalog key (then it is derived).",
    )
    p.add_argument(
        "--model",
        required=True,
        help="a catalog key (realtime_framework/providers/registry.py) OR a raw backend id / local model path",
    )
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument(
        "--model-input-fps",
        type=float,
        default=None,
        help="frames per second fed to the model (default: --fps, the environment's rate)",
    )
    p.add_argument(
        "--clock",
        choices=["wall", "token"],
        default=None,
        help="wall uses deployment time; token places actions at n/B frame positions.",
    )
    p.add_argument(
        "--tokens-per-frame",
        default=str(DEFAULT_TOKENS_PER_FRAME),
        help="positive token-to-time rate B per frame (default: %(default)s); "
        "'unlimited' acts at the observed tick, then advances one frame.",
    )
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="model-native reasoning effort; catalog models validate their own range.",
    )
    p.add_argument("--task", required=True, help="registered task id")
    p.add_argument("--seed", type=task_seed, default=os.environ.get("SEED", "42"))
    p.add_argument("--max-minutes", type=float, default=5.0)
    p.add_argument(
        "--think-budget-s",
        type=float,
        default=None,
        help="Deliberation clock: cumulative token-derived simulated seconds in token "
        "mode, wall seconds in wall mode. Legacy diagnostic only; the maintained "
        "protocol leaves this unset and terminates at T.",
    )
    p.add_argument(
        "--per-move-cap-s",
        type=float,
        default=None,
        help="Per-move deliberation cap in the active clock's seconds. A move over it "
        "times out (no action that turn; costs exactly the cap from the "
        "deliberation clock). Legacy diagnostic only; the maintained protocol "
        "leaves this unset.",
    )
    p.add_argument(
        "--protocol", default=None, help="Identity label for the executable evaluation protocol."
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--report-root", type=Path, help="Episode launch directory to validate and report")
    p.add_argument("--report-output", type=Path, help="Also write the episode report here")
    p.add_argument(
        "--graphics-mode",
        choices=("cpu", "gpu"),
        default=os.environ.get("SANDBOX_GRAPHICS_MODE", "cpu"),
        help="Desktop browser renderer: cpu uses SwiftShader; gpu uses NVIDIA Vulkan.",
    )
    p.add_argument(
        "--recording-encoder",
        choices=("cpu", "gpu"),
        default=os.environ.get("SCREEN_RECORDING_ENCODER", "cpu"),
        help="Screen recording encoder: cpu uses libx264; gpu uses NVENC.",
    )
    a = p.parse_args()
    os.environ["SANDBOX_GRAPHICS_MODE"] = a.graphics_mode
    os.environ["SCREEN_RECORDING_ENCODER"] = a.recording_encoder
    a.mode = DEFAULT_MODE  # fixed identity field; the runner has one decision loop
    a.clock = a.clock or "token"
    if a.clock == "wall" or a.tokens_per_frame == "unlimited":
        tokens_per_frame = None
    else:
        try:
            tokens_per_frame = int(a.tokens_per_frame)
        except ValueError:
            p.error("--tokens-per-frame must be a positive integer or 'unlimited'")
        if tokens_per_frame <= 0:
            p.error("--tokens-per-frame must be positive")
    protocol = a.protocol or ("wall-clock" if a.clock == "wall" else "time-aware-token-v1")
    if not 0 < a.fps < float("inf") or not 0 < a.max_minutes < float("inf"):
        p.error("--fps and --max-minutes must be finite and positive")
    if a.model_input_fps is None:
        a.model_input_fps = a.fps  # prepare_episode_report reads it from the arguments
    if not 0 < a.model_input_fps < float("inf"):
        p.error("--model-input-fps must be finite and positive")
    # --fps is the environment's frame rate (task clock, ticks, loop, scored world time);
    # --model-input-fps the rate of the frames the model is fed (ModelConfig.fps, B, the prompt).
    env_fps, model_input_fps = a.fps, a.model_input_fps
    if protocol == "time-aware-token-v1":
        if a.clock != "token":
            p.error("time-aware-token-v1 requires --clock token")
        if a.think_budget_s is not None or a.per_move_cap_s is not None:
            p.error(
                "time-aware-token-v1 uses only the task horizon T; deliberation caps are "
                "legacy diagnostics"
            )
    requested_model_identity = _model_identity(a.model, a.adapter)
    catalog_entry = catalog.get(a.model) if catalog.has(a.model) else None
    if catalog_entry is not None:
        reasoning_effort = (
            a.reasoning_effort
            if a.reasoning_effort is not None
            else catalog_entry.defaults["reasoning_effort"]
        )
        try:
            catalog.validate_reasoning_effort(a.model, reasoning_effort)
        except ValueError as exc:
            p.error(str(exc))
    else:
        reasoning_effort = a.reasoning_effort or "low"
    benchmark_cell_eligible = bool(
        catalog_entry.benchmark_eligible if catalog_entry is not None else True
    )
    comparison_scope = catalog_entry.comparison_scope if catalog_entry is not None else "ad_hoc"
    if not benchmark_cell_eligible and os.environ.get("ALLOW_EXPERIMENTAL_MODEL") != "1":
        p.error(
            f"catalog cell {a.model!r} is excluded from maintained benchmark results; "
            "set ALLOW_EXPERIMENTAL_MODEL=1 only for explicitly non-comparable diagnostics"
        )
    try:
        check_token_budget(
            requested_model_identity["adapter"],
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
        )
    except UnsupportedTokenBudget as exc:
        p.error(str(exc))
    run_id = a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(a.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id=run_id)
    episode_trace.emit(
        TraceEvent.EPISODE_STARTED,
        {
            "task_id": a.task,
            "requested_model": requested_model_identity,
            "clock_mode": a.clock,
            "mode": a.mode,
            "fps": env_fps,
            "model_input_fps": model_input_fps,
            "tokens_per_frame": tokens_per_frame,
            "reasoning_effort": reasoning_effort,
            "max_minutes": a.max_minutes,
            "seed": a.seed,
            "protocol": protocol,
            "benchmark_eligible": benchmark_cell_eligible,
            "comparison_scope": comparison_scope,
            "graphics_mode": a.graphics_mode,
            "recording_encoder": a.recording_encoder,
        },
    )

    env_config = EnvironmentConfig()
    R.build_image_if_needed()
    if a.report_root:
        from realtime_framework.sandbox.apptainer import build_image
        from realtime_framework.evaluation.reporting import prepare_episode_report

        build_image("realtime")
        prepare_episode_report(a)
    R.start_container(env_config)
    guest = GuestClient(host="localhost", port=env_config.guest_agent_port)
    if not guest.wait_until_ready(max_wait=60):
        raise RuntimeError("guest not ready")
    R._start_screensaver_keepalive()
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Initialize bookkeeping before setup so early failures produce a summary
    # and the finally block can release resources.
    ctask = None
    task = None
    agent = None
    agent_trace = None
    cfg = None
    env = None
    think_budget = None
    per_move_cap = None
    frozen_snapshot = None
    timeline = []
    verification = None
    solved = None
    error = None
    generation_artifact = None
    artifact_error = None
    evaluation_path = None
    environment_invalid = False
    invalid_run = None
    scene_capture = None
    tool_runtime = None
    loop_time_s = None
    _recorder_on = False
    active_loop = None
    harness_hash = harness_fingerprint()
    runtime_config = None
    runtime_config_hash = None
    t0 = time.time()
    run_stage = "environment_setup"
    try:
        # -------- provision the registered task --------
        ctask = custom_tasks.get(a.task)
        ctask.configure_seed(a.seed)
        task = ctask.description()
        ctask.configure_clock(a.clock, env_fps, tokens_per_frame)
        url = ctask.provision()
        if url:
            R.open_local_url_in_container(url, debug_port=getattr(ctask, "dbg_port", None))
        time.sleep(5)

        # -------- resolve the model + build the agent --------
        # A catalog key (e.g. gpt-realtime-2, claude-opus-5) sets the adapter +
        # backend id/path + per-model defaults from realtime_framework/providers/registry.py; a raw id/path still works
        # if --adapter is given. B controls simulated time rather than response length; only the
        # remaining episode horizon and native model limits can bound a decision.
        run_stage = "provider_setup"
        if catalog.has(a.model):
            cfg = catalog.to_model_config(
                a.model,
                fps=model_input_fps,
                reasoning_effort=reasoning_effort,
            )
        else:
            if not a.adapter:
                p.error("--adapter is required when --model is not a catalog key")
            cfg = ModelConfig(
                adapter_type=a.adapter,
                model_name=a.model,
                fps=model_input_fps,
                reasoning_effort=reasoning_effort,
            )
        setup = AgentSetup()
        try:
            set_up_agent(
                setup,
                cfg,
                ctask,
                task,
                clock_mode=a.clock,
                tokens_per_frame=tokens_per_frame,
                run_dir=run_dir,
                episode_trace=episode_trace,
                model=requested_model_identity["model"],
                open_capture=lambda text_observation: BrowserSceneCapture(
                    guest,
                    ctask,
                    debug_port=getattr(ctask, "dbg_port", None),
                    container=R.CONTAINER_NAME,
                    desktop_size=(env_config.screen_width, env_config.screen_height),
                    text_observation=text_observation,
                ),
            )
        finally:
            # The cleanup below closes what setup made, also when one of its steps failed.
            run_stage, environment_invalid = setup.stage, setup.environment_invalid
            agent, agent_trace = setup.agent, setup.agent_trace
            scene_capture, tool_runtime = setup.scene_capture, setup.tool_runtime

        # Optional legacy diagnostics may add deliberation caps explicitly. The maintained method
        # uses only the shared episode horizon T and therefore does not inherit task-local caps.
        think_budget = a.think_budget_s
        if think_budget is not None:
            print(
                f"[think] deliberation budget = {think_budget:g}s of cumulative model thinking",
                flush=True,
            )
        per_move_cap = a.per_move_cap_s
        if per_move_cap is not None:
            print(
                f"[think] per-move cap = {per_move_cap:g}s (a slower move times out -> no action that turn)",
                flush=True,
            )
        runtime_config = effective_run_config(
            seed=a.seed,
            model_config=cfg,
            fps=env_fps,
            mode=a.mode,
            max_minutes=a.max_minutes,
            protocol=protocol,
            think_budget_s=think_budget,
            per_move_cap_s=per_move_cap,
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
        )
        runtime_config_hash = config_fingerprint(runtime_config)
        _atomic_write_json(
            run_dir / "run_config.json",
            {
                **runtime_config,
                "harness_version": harness_version(harness_hash),
                "harness_hash": harness_hash,
                "runtime_config_hash": runtime_config_hash,
            },
        )

        # initialize() may open a realtime session or prefill a local model.  Restart time-sensitive
        # wall-clock tasks only after that work is complete.  The restart is two-phase: reset_episode()
        # prepares a held episode (reload, baseline, semantics, browser render) and start_episode()
        # releases it.  The origin start_episode() reports is the scored wall-clock origin, so no
        # framework setup consumes the episode.  Record only after provider/environment
        # initialization; the first-frame PTS and explicit active boundaries remove the small
        # recorder startup margin.
        _recorder_on = R._start_screen_recording(run_dir)
        wall_start_time = None
        episode_start = None
        if (
            a.clock == "wall"
            and ctask is not None
            and hasattr(ctask, "reset_episode")
            and getattr(ctask, "category", None) in {"browser_game", "realtime"}
        ):
            category = getattr(ctask, "category", None)
            try:
                _rb = ctask.reset_episode()
                print(
                    f"[{category}] scored wall episode reset after adapter initialization; "
                    f"baseline={_rb}",
                    flush=True,
                )
                if category == "realtime":
                    # Let the open page render the held, restarted state before the clock starts.
                    time.sleep(0.45)
                episode_start = ctask.start_episode()
                wall_start_time = float(episode_start["clock_origin"])
                print(f"[{category}] scored wall episode started; {episode_start}", flush=True)
            except Exception:
                detail = traceback.format_exc()
                environment_invalid = True
                print(
                    f"[{category}] scored wall restart FAILED; run is invalid:\n" + detail,
                    flush=True,
                )
                raise EnvironmentInvalidError(
                    f"{category} episode restart failed after adapter initialization\n" + detail
                )

        # -------- environment budget --------
        if a.clock == "token":
            deadline = None
        else:
            wall_start_time = wall_start_time or time.time()
            deadline = wall_start_time + a.max_minutes * 60
        # Game tasks ignore model failure declarations. Any task may expose episode_over for a
        # server-owned terminal state; realtime tasks use it for irreversible failures.
        is_game = (getattr(ctask, "category", None) == "browser_game") if ctask else False
        env = GuestEnvironment(
            guest,
            deadline,
            ctask,
            frames_dir=frames_dir,
            honor_fail=(not is_game),
            stop_check=(getattr(ctask, "episode_over", None) if ctask else None),
            scene_capture=scene_capture,
            action_audit_path=run_dir / "action_lifecycle.jsonl",
            clock_mode=a.clock,
            wall_start_time=wall_start_time,
        )
        start_episode = getattr(agent, "start_episode", None)
        if callable(start_episode):
            start_episode(wall_start_time, a.max_minutes * 60)

        # -------- run the streaming loop --------
        _loop_t0 = time.time()
        run_stage = "inference"
        _active_start = wall_start_time if a.clock == "wall" else _loop_t0
        episode_trace.emit(
            "ActiveLoopStarted",
            {
                "start_unix_s": _active_start,
                "loop_entered_unix_s": _loop_t0,
                "deadline_unix_s": deadline,
                "clock_mode": a.clock,
                "episode_start": episode_start,
            },
        )
        try:
            timeline = run_stream(
                agent,
                env,
                delta_s=1.0 / env_fps,
                frame_ticks=env_fps / model_input_fps,
                max_ticks=a.max_minutes * 60 * env_fps,
                log=lambda m: print(m, flush=True),
                think_budget_s=think_budget,
                per_move_cap_s=per_move_cap,
                clock_mode=a.clock,
                tokens_per_frame=tokens_per_frame,
                trace=episode_trace,
            )
        finally:
            # The agent's run window is recorded even when the loop ends in an error.
            # Physical diagnostic only; the token-clock horizon is simulated and does not bound wall time.
            loop_time_s = round(time.time() - _loop_t0, 1)
            _active_end = getattr(env, "_active_end_unix_s", time.time())
            active_loop = {
                "start_unix_s": _active_start,
                "end_unix_s": _active_end,
                "loop_entered_unix_s": _loop_t0,
                "deadline_unix_s": deadline,
                "episode_start": episode_start,
                # Setup between the task's scored origin and the scheduler's first observation.
                "start_overhead_s": (_loop_t0 - wall_start_time) if a.clock == "wall" else 0.0,
                "active_wall_time_s": _active_end - _active_start,
                "clock_mode": a.clock,
                "horizon_s": a.max_minutes * 60,
            }
            _atomic_write_json(run_dir / "active-loop.json", active_loop)
            episode_trace.emit("ActiveLoopEnded", active_loop)
        R._stop_screen_recording(run_dir, frames_dir, model_input_fps)
        _recorder_on = False
        # INFERENCE/EVAL BOUNDARY: run_stream freezes live scorer input at the latched terminal tick,
        # before it releases held keys or mouse buttons. Consume that exact snapshot here, before
        # agent close or verifier work. The same value is stored in the artifact
        # so python -m realtime_framework.evaluation.reporting reevaluate agrees with the summary.
        if ctask is not None and hasattr(ctask, "evaluation_snapshot"):
            try:
                snapshot_error = getattr(env, "_termination_snapshot_error", None)
                if snapshot_error:
                    raise RuntimeError(snapshot_error)
                if getattr(env, "_termination_snapshot_captured", False):
                    frozen_snapshot = getattr(env, "_termination_snapshot", None)
                else:
                    # Compatibility fallback for an alternate environment implementation that does
                    # not expose the pre-cleanup snapshot contract.
                    frozen_snapshot = ctask.evaluation_snapshot()
                snapshot_invalid_reason = _invalid_snapshot(frozen_snapshot)
                if snapshot_invalid_reason:
                    invalid_run = {
                        "category": "environment",
                        "reason": snapshot_invalid_reason,
                        "detail": json.dumps(frozen_snapshot, default=str)[:2000],
                    }
                episode_trace.emit(
                    TraceEvent.SNAPSHOT_FROZEN,
                    {
                        "kind": (frozen_snapshot or {}).get("kind"),
                        "snapshot_sha256": canonical_sha256(frozen_snapshot),
                        "artifact": "generation_artifact/task_state.json",
                        "valid": snapshot_invalid_reason is None,
                        "invalid_reason": snapshot_invalid_reason,
                        "environment_tick": env.tick_index() if env is not None else None,
                    },
                )
                episode_trace.checkpoint()
            except Exception:
                detail = traceback.format_exc()
                invalid_run = {
                    "category": "environment",
                    "reason": "evaluation_snapshot_failed",
                    "detail": detail,
                }
                episode_trace.emit(TraceEvent.RUN_INVALIDATED, invalid_run)
                print("[invalid-run] evaluation snapshot failed:\n" + detail, flush=True)
        try:
            generation_artifact = capture_generation_artifact(
                run_dir,
                ctask=ctask,
                adapter=cfg.adapter_type,
                model=requested_model_identity["model"],
                backend_model=cfg.model_name,
                mode=a.mode,
                cap=a.max_minutes,
                protocol=protocol,
                harness_hash=harness_hash,
                fps=env_fps,
                model_input_fps=model_input_fps,
                runtime_config=runtime_config,
                runtime_config_hash=runtime_config_hash,
                snapshot=frozen_snapshot,
            )
            print("[artifact] generation state saved before eval", flush=True)
        except Exception:
            artifact_error = traceback.format_exc()
            print(
                "[artifact] capture failed (evaluation will continue):\n" + artifact_error,
                flush=True,
            )

        # A per-move generate the runner abandoned on timeout is still running (daemon, un-killable).
        # Wait it out HERE — after the evaluation snapshot and artifact are frozen, so the drain adds
        # no drift to the score, and before close(), which would otherwise tear the model/session down
        # underneath a live native call. Bounded; _hard_exit covers a worker that never returns.
        try:
            drain = getattr(env, "_drain_abandoned", None)
            if drain is not None:
                drain()
        except Exception:
            pass
        try:
            agent.close()
        except Exception:
            invalid_run = invalid_run or {
                "category": "provider",
                "reason": "provider_close_failed",
                "detail": traceback.format_exc(),
            }

        run_stage = "evaluation"
        try:
            if (
                ctask is not None
                and frozen_snapshot is not None
                and hasattr(ctask, "evaluate_snapshot")
            ):
                # score the FROZEN snapshot (== the artifact), not a fresh post-cleanup live read
                verification = ctask.evaluate_snapshot(frozen_snapshot)
            else:
                verification = ctask.verify()
            evaluation_invalid_reason = _invalid_evaluation(verification)
            if evaluation_invalid_reason:
                invalid_run = invalid_run or {
                    "category": "evaluator",
                    "reason": evaluation_invalid_reason,
                    "detail": json.dumps(verification, default=str)[:2000],
                }
                solved = None
            else:
                solved = verification["resolved"]
            evaluation_path = str(save_evaluation(run_dir, verification))
            episode_trace.emit(
                TraceEvent.SCORE_PRODUCED,
                {
                    "valid": evaluation_invalid_reason is None,
                    "invalid_reason": evaluation_invalid_reason,
                    "resolved": solved,
                    "verification": verification,
                    "artifact": "evaluation.json",
                },
            )
            episode_trace.checkpoint()
        except Exception:
            _eval_error = traceback.format_exc()
            invalid_run = invalid_run or {
                "category": "evaluator",
                "reason": "evaluation_exception",
                "detail": _eval_error,
            }
            episode_trace.emit(TraceEvent.RUN_INVALIDATED, invalid_run)
            evaluation_path = str(save_evaluation(run_dir, None, error=_eval_error))
            print("[invalid-run] evaluator exception:\n" + _eval_error, flush=True)
    except Exception:
        error = traceback.format_exc()
        invalid_run = (
            getattr(env, "_invalid_run", None) if env is not None else None
        ) or invalid_run
        invalid_run = invalid_run or {
            "category": (
                "environment"
                if isinstance(sys.exc_info()[1], EnvironmentInvalidError)
                or run_stage == "environment_setup"
                else "provider"
                if run_stage == "provider_setup"
                else "tool"
                if run_stage == "tool_setup"
                else "evaluator"
                if run_stage == "evaluation"
                else "runner"
            ),
            "reason": "environment_preflight_failed" if environment_invalid else "run_exception",
            "detail": error,
        }
        print("RUN ERROR\n" + error, flush=True)
        if evaluation_path is None:
            try:
                evaluation_path = str(save_evaluation(run_dir, None, error=error))
            except Exception:
                pass
    finally:
        # Stop recording and close model resources even when setup or scoring fails.
        # The shared container remains available for subsequent runs.
        if _recorder_on:
            try:
                R._stop_screen_recording(run_dir, frames_dir, model_input_fps)
            except Exception:
                pass
        try:
            # Same reason as the main path: never close() under a live abandoned native call.
            # Idempotent, so this is a no-op when the main path already drained. env is None if we
            # failed before the loop existed, in which case there is nothing to drain.
            drain = getattr(env, "_drain_abandoned", None)
            if drain is not None:
                drain()
        except Exception:
            pass
        try:
            if agent is not None:
                agent.close()
        except Exception:
            invalid_run = invalid_run or {
                "category": "provider",
                "reason": "provider_close_failed",
                "detail": traceback.format_exc(),
            }
        # After the agent is closed: stop new agent records, write the queued ones and fence the
        # writer before the checkpoint below and EpisodeFinished.
        if agent_trace is not None:
            invalid_run = close_agent_trace(agent_trace, invalid_run)
        try:
            if tool_runtime is not None:
                tool_runtime.close()
        except Exception:
            invalid_run = invalid_run or {
                "category": "tool",
                "reason": "tool_trace_close_failed",
                "detail": traceback.format_exc(),
            }
        try:
            if scene_capture is not None:
                scene_capture.close()
        except Exception:
            invalid_run = invalid_run or {
                "category": "environment",
                "reason": "cdp_close_failed",
                "detail": traceback.format_exc(),
            }
        try:
            episode_trace.checkpoint()
        except Exception:
            invalid_run = invalid_run or {
                "category": "harness",
                "reason": "episode_trace_checkpoint_failed",
                "detail": traceback.format_exc(),
            }
    deltas = [x[2] for x in timeline]
    # Billing uses complete provider-reported cost when available (OpenRouter), otherwise a
    # versioned catalog rate card with modality and cache-read splits (direct providers).
    tok = getattr(agent, "token_usage", None) if agent is not None else None
    adapter_tok = tok
    tok = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "responses": 0}
    accounting = getattr(env, "_token_accounting", None) if env is not None else None
    token_accounting = accounting.summary() if accounting is not None else None
    if token_accounting is not None:
        counts = token_accounting["counts"]
        tok = {
            name: counts["provider_" + name]["total"]
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        tok["responses"] = token_accounting["responses"]
    cost_accounting = summarize_cost(
        token_accounting,
        pricing=(catalog_entry.pricing if catalog_entry is not None else None),
        provider=(catalog_entry.provider if catalog_entry is not None else None),
    )
    from realtime_framework.evaluation.clock import summarize_model_timing

    model_timing = summarize_model_timing(run_dir)
    cost_usd = cost_accounting.get("total_usd") if cost_accounting.get("complete") else None
    cost_known_usd = cost_accounting.get("known_cost_usd")
    invalid_run = (getattr(env, "_invalid_run", None) if env is not None else None) or invalid_run
    if environment_invalid:
        invalid_run = invalid_run or {
            "category": "environment",
            "reason": "environment_preflight_failed",
        }
    if invalid_run is not None:
        solved = None
    run_status = (
        "invalid"
        if invalid_run is not None
        else ("scored" if benchmark_cell_eligible else "experimental")
    )
    # The stop reason and the end time are LATCHED by the runner at loop exit (env._stop_reason,
    # env._termination), so cleanup that crosses the deadline rewrites neither a real environment
    # terminal state into time_limit nor the episode's length. Any invalid run reports
    # "invalid_run"; "setup_error" if we never reached the loop.
    latched = getattr(env, "_stop_reason", None) if env is not None else None
    termination = getattr(env, "_termination", None) if env is not None else None
    stop_reason = (
        "invalid_run"
        if invalid_run is not None
        else (latched or ("setup_error" if env is None else "unknown"))
    )
    world_time_s = termination["end_s"] if termination is not None else None
    try:
        run_metrics = summarize_run_metrics(
            run_dir / "action_lifecycle.jsonl",
            total_decisions=len(timeline),
            solved=solved,
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
            fps=model_input_fps,
            termination=termination,
            horizon_s=a.max_minutes * 60,
        )
    except Exception:
        metrics_error = traceback.format_exc()
        invalid_run = invalid_run or {
            "category": "harness",
            "reason": "run_metrics_aggregation_failed",
            "detail": metrics_error,
        }
        solved = None
        run_status = "invalid"
        stop_reason = "invalid_run"
        run_metrics = summarize_run_metrics(
            run_dir / "missing-action-lifecycle.jsonl",
            total_decisions=len(timeline),
            solved=None,
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
            fps=model_input_fps,
            termination=None,
            horizon_s=a.max_minutes * 60,
            validate_decision_count=False,
        )
        run_metrics["run_metrics_error"] = metrics_error
    effective_model_identity = {
        "adapter": cfg.adapter_type if cfg else requested_model_identity["adapter"],
        "model": requested_model_identity["model"],
        "backend_model": cfg.model_name if cfg else requested_model_identity["backend_model"],
    }
    run_identity = resolve_run_identity(
        task_id=a.task,
        ctask=ctask,
        **effective_model_identity,
        mode=a.mode,
        cap=a.max_minutes,
        protocol=protocol,
        harness_hash=harness_hash,
        fps=env_fps,
        model_input_fps=model_input_fps,
        runtime_config_hash=runtime_config_hash,
    )
    summ = {
        **run_identity,
        "instance_id": a.task,
        "task_description": task,
        "reasoning_effort": reasoning_effort,
        "benchmark_eligible": benchmark_cell_eligible,
        "comparison_scope": comparison_scope,
        "interactables_protocol": (INTERACTABLES_PROTOCOL if tool_runtime is not None else None),
        "interactables_schema": (INTERACTABLES_SCHEMA if tool_runtime is not None else None),
        "interactables_representation": (
            INTERACTABLES_REPRESENTATION if tool_runtime is not None else None
        ),
        "interactables_artifact": ("tool_calls.jsonl" if tool_runtime is not None else None),
        "run_status": run_status,
        "invalid_run": invalid_run,
        "solved": solved,
        "decisions": len(timeline),
        "actions_executed": getattr(env, "actions", 0) if env is not None else 0,
        "actions_failed": getattr(env, "failed", 0) if env is not None else 0,
        "avg_delta_ticks": round(sum(deltas) / len(deltas), 2) if deltas else None,
        "token_usage": tok,
        "cost_usd": cost_usd,
        "cost_known_usd": cost_known_usd,
        "cost_status": cost_accounting.get("status"),
        "cost_source": cost_accounting.get("source"),
        "cost_accounting": cost_accounting,
        "token_usage_source": "provider_reported_inference_responses",
        "token_accounting": token_accounting,
        "token_clock_sources": (
            token_accounting.get("clock_sources") if token_accounting is not None else None
        ),
        "adapter_token_usage": adapter_tok,
        "adapter_token_usage_note": "Adapter-native cumulative counters; not used for the action clock or billing.",
        "token_accounting_source": (
            ",".join(token_accounting.get("clock_sources") or []) or None
            if token_accounting is not None
            else None
        ),
        "stop_reason": stop_reason,
        "failure_attribution": ((invalid_run or {}).get("category") if invalid_run else None),
        "environment_valid": (
            False
            if (invalid_run or {}).get("category") == "environment" or environment_invalid
            else None
        ),
        "loop_stop_reason": latched,
        "termination": termination,
        "max_minutes": a.max_minutes,
        "seed": a.seed,
        "applied_seed": getattr(ctask, "applied_seed", None),
        "clock_mode": a.clock,
        "tokens_per_frame": tokens_per_frame,
        "dropped_intermediate_frames": int(getattr(env, "_dropped_intermediate_frames", 0) or 0)
        if env is not None
        else 0,
        "output_clock_policy": OUTPUT_CLOCK_POLICY,
        "tokens_per_world_second": (
            tokens_per_frame * model_input_fps
            if a.clock == "token" and tokens_per_frame is not None
            else None
        ),
        "world_time_s": world_time_s,
        "last_clock_advance": (
            getattr(ctask, "_last_clock_advance", None) if ctask is not None else None
        ),
        "think_budget_s": think_budget,
        "per_move_cap_s": per_move_cap,
        "think_spent_s": round(getattr(env, "_think_spent", 0.0), 1) if env is not None else 0.0,
        "runtime_config": runtime_config,
        "observation_stream_frozen": bool(getattr(env, "_stream_frozen", False))
        if env is not None
        else False,
        "observation_stream_freeze_count": int(getattr(env, "_stream_freeze_count", 0) or 0)
        if env is not None
        else 0,
        "loop_time_s": loop_time_s,
        "active_wall_time_s": (active_loop or {}).get("active_wall_time_s"),
        "active_loop_artifact": "active-loop.json"
        if (run_dir / "active-loop.json").is_file()
        else None,
        "captures": len(getattr(env, "capture_times", [])) if env is not None else 0,
        "effective_capture_fps": _effective_capture_fps(env),
        "avg_capture_latency_s": _metric_avg(env, "capture_latencies"),
        "max_capture_latency_s": _metric_max(env, "capture_latencies"),
        "avg_frame_age_at_action_s": _metric_avg(env, "frame_ages"),
        "max_frame_age_at_action_s": _metric_max(env, "frame_ages"),
        "action_lifecycle_artifact": (
            "action_lifecycle.jsonl" if (run_dir / "action_lifecycle.jsonl").is_file() else None
        ),
        "wall_time_s": model_timing["wall_time_s"],
        "model_timing": model_timing,
        "total_run_time_s": round(time.time() - t0, 1),
        "verification": verification,
        "error": error,
        "generation_artifact": (
            "generation_artifact/manifest.json" if generation_artifact is not None else None
        ),
        "artifact_error": artifact_error,
        "evaluation_artifact": ("evaluation.json" if evaluation_path is not None else None),
        # Partial-credit progress headline (None for games without partial credit).
        "progress": (verification or {}).get("progress"),
        **run_metrics,
    }
    if invalid_run is not None:
        episode_trace.emit(TraceEvent.RUN_INVALIDATED, invalid_run)
    episode_trace.emit(
        TraceEvent.EPISODE_FINISHED,
        {
            "run_status": run_status,
            **episode_outcome(summ),
            "stop_reason": stop_reason,
            "resolved": solved,
            "decisions": len(timeline),
            "actions_executed": getattr(env, "actions", 0) if env is not None else 0,
            "world_time_s": world_time_s,
        },
    )
    summ.update(episode_outcome(summ))
    summ["episode_trace"] = episode_trace.describe()
    episode_trace.close()
    _atomic_write_json(run_dir / "run_summary.json", summ)
    from realtime_framework.evaluation.reporting import finalize_video

    finalize_video(run_dir)
    print(
        "SUMMARY",
        json.dumps(
            {
                k: summ[k]
                for k in [
                    "adapter",
                    "mode",
                    "run_status",
                    "outcome",
                    "error_reason",
                    "solved",
                    "progress",
                    "decisions",
                    "actions_executed",
                    "actions_failed",
                    "avg_delta_ticks",
                    "stop_reason",
                ]
            }
        ),
    )
    # Infrastructure-invalid runs are never successful process results and are never resumable as
    # scored cells. The final tasks suite additionally requires a Boolean score for every valid run.
    accepted = run_status == "scored" or (
        run_status == "experimental" and os.environ.get("ALLOW_EXPERIMENTAL_MODEL") == "1"
    )
    if not accepted or (custom_tasks.suite_id() == "tasks" and not isinstance(solved, bool)):
        raise SystemExit(1)


class AgentSetup:
    """What set_up_agent() has made so far. main() closes these after the episode, also when a
    later setup step fails, so each is recorded as soon as it exists. ``stage`` is main()'s
    run_stage, which attributes a failure, and ``environment_invalid`` marks a failed check of
    the environment."""

    def __init__(self):
        self.stage = "provider_setup"
        self.environment_invalid = False
        self.agent_trace = None
        self.agent = None
        self.scene_capture = None
        self.tool_runtime = None


def set_up_agent(
    setup: AgentSetup,
    cfg: ModelConfig,
    ctask,
    instruction: str,
    *,
    clock_mode: str,
    tokens_per_frame: int | None,
    run_dir: Path,
    episode_trace: EpisodeTrace,
    model: str,
    open_capture,
) -> None:
    """Turn a model config and a provisioned task into an agent initialized on the first frame.

    ``instruction`` is the task description and ``model`` the model identity the tool trace
    records. ``open_capture(text_observation)`` returns the scene capture for the observation
    form the model's context policy names. What this makes is recorded in ``setup``.
    """
    # Every pre-run rule on the model, clock and task, before any of them is built.
    check_episode(cfg, clock_mode=clock_mode, task=ctask)
    # The agent borrows the episode trace's write entry; the runner still opens and closes it.
    setup.agent_trace = AgentTraceSink(episode_trace, run_dir / "episode_blobs.pack")
    agent = setup.agent = create_agent(cfg, log_dir=run_dir, trace=setup.agent_trace)
    configure_agent(
        agent,
        fps=cfg.fps,
        tokens_per_frame=tokens_per_frame,
        clock_mode=clock_mode,
        category=getattr(ctask, "category", None),
    )
    setup.stage = "environment_setup"

    # Prepare a valid provisional episode/frame for adapter initialization. main() restarts
    # wall-clock tasks once more after initialize(), so none of this warmup consumes their scored
    # action window.
    if hasattr(ctask, "reset_episode"):
        category = getattr(ctask, "category", None)
        if category in {"browser_game", "realtime"}:
            try:
                _rb = ctask.reset_episode()
                print(f"[{category}] fresh episode reset; baseline={_rb}", flush=True)
                if category == "realtime" and clock_mode != "token":
                    time.sleep(0.45)
            except Exception:
                detail = traceback.format_exc()
                setup.environment_invalid = True
                print(f"[{category}] fresh restart FAILED; run is invalid:\n" + detail, flush=True)
                raise EnvironmentInvalidError(
                    f"{category} episode restart failed before tick 0\n" + detail
                )
    ctask.activate_clock()
    print(f"[clock] {ctask.clock_snapshot()}", flush=True)
    setup.stage = "tool_setup"
    # Choice models need focused fields and control state to enumerate actions,
    # including when they also receive screenshots (context policy item 12).
    context_policy = phase1_context_policy().for_path(cfg.adapter_type, cfg.provider_options)
    text_observation = context_policy.observation != "som_screenshot"
    scene_capture = setup.scene_capture = open_capture(text_observation)
    setup.tool_runtime = ToolRuntime(
        run_dir / "tool_calls.jsonl",
        task_id=ctask.task_id,
        adapter=cfg.adapter_type,
        model=model,
        text_observation=text_observation,
    )
    if hasattr(agent, "bind_tool_runtime"):
        agent.bind_tool_runtime(setup.tool_runtime)
    setup.stage = "environment_setup"
    # Chrome can start with keyboard focus in its address bar; the model acts on the page.
    try:
        scene_capture.ensure_page_focus()
    except Exception as exc:
        setup.environment_invalid = True
        raise EnvironmentInvalidError(f"page did not take keyboard focus: {exc}") from exc
    initial_observation = scene_capture.capture(0)
    initial_interactables = initial_observation.metadata.get("interactables") or {}
    if initial_interactables.get("scene", {}).get(
        "status"
    ) != "ok" or not initial_interactables.get("binding", {}).get("stable"):
        setup.environment_invalid = True
        raise EnvironmentInvalidError(
            "initial observation has unavailable or unstable interactables: "
            + json.dumps(initial_interactables, default=str)[:1000]
        )
    setup.stage = "provider_setup"
    agent.initialize(instruction, initial_observation)
    # Initialization is allowed to open a native session or warm a local KV cache, but the
    # scored first turn receives exactly one freshly bound observation from the frame clock.
    agent.discard_pending_observations()
    setup.stage = "environment_setup"


def _write_emergency_setup_summary(error: str):
    """Persist setup/provision failures that happen before main() enters its episode try/finally."""

    def arg(name, default=None):
        try:
            return sys.argv[sys.argv.index(name) + 1]
        except (ValueError, IndexError):
            return default

    run_id = arg("--run-id")
    results_dir = arg("--results-dir", "results")
    if not run_id:
        return
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_summary.json"
    if path.exists():
        return
    requested_model = arg("--model")
    model_identity = _model_identity(requested_model, arg("--adapter"))
    mode = DEFAULT_MODE
    clock_mode = arg("--clock", "token")
    protocol = arg(
        "--protocol",
        "wall-clock" if clock_mode == "wall" else "time-aware-token-v1",
    )
    cap_minutes = float(arg("--max-minutes", "5.0"))
    env_fps = float(arg("--fps", "1.0"))
    model_input_fps = float(arg("--model-input-fps", str(env_fps)))
    run_identity = resolve_run_identity(
        task_id=arg("--task"),
        **model_identity,
        mode=mode,
        cap=cap_minutes,
        protocol=protocol,
        harness_hash=harness_fingerprint(),
        fps=env_fps,
        model_input_fps=model_input_fps,
    )
    tokens_per_frame_arg = arg("--tokens-per-frame", str(DEFAULT_TOKENS_PER_FRAME))
    tokens_per_frame = (
        None
        if clock_mode == "wall" or tokens_per_frame_arg == "unlimited"
        else int(tokens_per_frame_arg)
    )
    run_metrics = summarize_run_metrics(
        run_dir / "action_lifecycle.jsonl",
        total_decisions=0,
        solved=None,
        clock_mode=clock_mode,
        tokens_per_frame=tokens_per_frame,
        fps=model_input_fps,
        termination=None,
        horizon_s=cap_minutes * 60,
    )
    cost_accounting = summarize_cost(None)
    _atomic_write_json(
        path,
        {
            **run_identity,
            "instance_id": arg("--task"),
            "max_minutes": cap_minutes,
            "seed": task_seed(arg("--seed", os.environ.get("SEED", "42"))),
            "clock_mode": clock_mode,
            "tokens_per_frame": tokens_per_frame,
            "world_time_s": None,
            "run_status": "invalid",
            "invalid_run": {"category": "runner", "reason": "setup_error", "detail": error},
            "outcome": "error",
            "error_category": "runner",
            "error_reason": "setup_error",
            "failure_attribution": "runner",
            "solved": None,
            "decisions": 0,
            "actions_executed": 0,
            "stop_reason": "setup_error",
            "loop_stop_reason": None,
            "termination": None,
            "verification": None,
            "error": error,
            "cost_usd": None,
            "cost_known_usd": None,
            "cost_status": cost_accounting["status"],
            "cost_source": cost_accounting["source"],
            "cost_accounting": cost_accounting,
            **run_metrics,
        },
    )


def _hard_exit(code: int):
    """Exit without finalizing Python while timed-out native workers may still run.

    The normal teardown first drains workers within a bounded interval and
    persists results. This fallback avoids unwinding live native calls during
    interpreter shutdown; flush remaining output before exiting."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        logging.shutdown()  # flush configured log handlers
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    # The episode entry point execs this process, so interrupts reach the runner directly.
    _report_parser = argparse.ArgumentParser(add_help=False)
    _report_parser.add_argument("--report-root", type=Path)
    _report_parser.add_argument("--report-output", type=Path)
    _report_args, _ = _report_parser.parse_known_args()
    _interrupted_signal = 2
    if _report_args.report_root:
        import signal

        def _interrupt_episode(signum, _frame):
            global _interrupted_signal
            _interrupted_signal = signum
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _interrupt_episode)
    _code = 0
    try:
        main()
    except SystemExit as e:  # main() raises SystemExit(1) for an unscored benchmark cell
        _code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:
        # main() ran its finalizers; an interrupted episode is never a scored result.
        print(
            "INTERRUPTED — episode did not complete",
            flush=True,
        )
        _code = 128 + _interrupted_signal
    except Exception:
        _error = traceback.format_exc()
        print("FATAL SETUP/RUN ERROR\n" + _error, flush=True)
        try:
            _write_emergency_setup_summary(_error)
        except Exception:
            print("Could not write emergency run summary:\n" + traceback.format_exc(), flush=True)
        _code = 1
    finally:
        if _report_args.report_root:
            try:
                R.stop_containers()
            except Exception:
                print("DESKTOP CLEANUP ERROR\n" + traceback.format_exc(), flush=True)
                _code = _code or 1
            try:
                from realtime_framework.evaluation.reporting import finish_episode_report

                _report_code = finish_episode_report(
                    _report_args.report_root, _report_args.report_output
                )
                _code = _code or _report_code
            except Exception:
                print("REPORT ERROR\n" + traceback.format_exc(), flush=True)
                _code = _code or 1
    # os._exit rather than re-raising: see _hard_exit. The traceback above is already printed in
    # full, so routing it through the default excepthook would only duplicate it.
    _hard_exit(_code)
