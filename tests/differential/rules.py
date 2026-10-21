"""The expected differences between today's loop and the kernel, as numbered rewrite rules
(wave2-plan.md, 5 预期差异清单; wave2-overrides.md, X12 and X13).

Each rule rewrites the old side's projection into what the new side shows; the rules run in
order and what is left must match exactly. A rule triggers when it changed something. A script's
``expect`` names the rules it may trigger: one it did not name fails the comparison, one it named
that did not trigger is reported. Rules without a rewrite cannot show between today's loop and the
simulation; ``seen`` says where they can. U-numbered rules are differences this framework found
that the list does not have yet.

The old side's times come from its facts: when each generation started and returned, when each
decision was resolved and committed, and every end check with its answer. The wall-clock rules
assume the script's operations take no time, which the random scripts keep; named scripts that
give them time name the rules they show.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .projection import TOLERANCE, Projection, differences, earlier, key_releases
from .script import Script


@dataclass(frozen=True)
class Rule:
    id: str
    summary: str
    rewrite: Callable[[Script, Projection, dict], None] | None = None
    seen: str = "today's loop against the simulation"


@dataclass
class Verdict:
    script: Script
    triggered: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    close: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.unexpected


def check(script: Script, old: Projection, new: Projection) -> Verdict:
    current = old.copy()
    notes: dict = {}
    verdict = Verdict(script)
    for rule in RULES:
        if rule.rewrite is None:
            continue
        before = copy.deepcopy(current.compared)
        rule.rewrite(script, current, notes)
        if current.compared != before:
            verdict.triggered.append(rule.id)
    verdict.problems, verdict.close = differences(current.compared, new.compared)
    if verdict.close and "X4" not in verdict.triggered:
        verdict.triggered.append("X4")
    verdict.triggered.sort(key=ORDER.index)
    verdict.unexpected = [rule for rule in verdict.triggered if rule not in script.expect]
    verdict.unused = sorted(set(script.expect) - set(verdict.triggered), key=ORDER.index)
    return verdict


# ---- helpers ----------------------------------------------------------------------------------


def _wall(script: Script) -> bool:
    return script.clock == "wall"


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE


def _floor(value: float) -> int:
    return math.floor(value + 1e-9)


def _frames(script: Script, ticks: float) -> float:
    return ticks * script.input_fps / script.env_fps


def _dropped(script: Script, t_k: float, reached: float) -> int:
    """Input frames a decision crossed but one (runner.py:664-665)."""
    crossed = max(0, _floor(_frames(script, reached)) - _floor(_frames(script, t_k)))
    return max(0, crossed - 1)


def _give_up(p: Projection, k: int) -> None:
    """Decision k's generation was given up on: it keeps its observation and nothing else, and
    the agent is aborted, unless today's loop aborted it too (its bounded wait ran out)."""
    started = p.facts["started_at"][k]
    if not any(not earlier(at, started) for at in p.facts["aborted_at"]):
        p.episode["aborts"] += 1
    d = p.decisions[k]
    if d["commit"] is not None:
        p.episode["snapshot_after"] -= 1
    for name in (
        "proposed",
        "approach",
        "resolved",
        "wait_record",
        "boundary",
        "commit",
        "outcome",
    ):
        d[name] = None
    d["applied"], d["executed"] = False, []


def _drop_after(p: Projection, k: int) -> None:
    for later in [j for j in p.decisions if j > k]:
        if p.decisions[later]["commit"] is not None:
            p.episode["snapshot_after"] -= 1
        del p.decisions[later]


def _grid_check(script: Script, at: float) -> float | None:
    """The first periodic wall end check at or after ``at``: the kernel asks at j / env_fps,
    j >= 1, while the episode lasts (turns.py:229-233, 659-663)."""
    fps = script.env_fps
    j = max(1, math.ceil(at * fps))
    while j > 1 and (j - 1) / fps >= at:
        j -= 1
    while j / fps < at:
        j += 1
    return j / fps if j / fps < script.duration_s else None


def _in_hand_at(p: Projection, at: float) -> int | None:
    """The old side's decision in hand at ``at``: started before it, committed at or after."""
    facts = p.facts
    for k in sorted(facts["started_at"]):
        committed = facts["committed_at"].get(k)
        if earlier(facts["started_at"][k], at) and (
            committed is None or not earlier(committed, at)
        ):
            return k
    return None


# ---- the rules --------------------------------------------------------------------------------


def _x4(script: Script, p: Projection, notes: dict) -> None:
    """Today's wall tick is int(clock position) (sandbox/actions.py:200-203); the kernel floors the
    position plus 1e-9 (turns.py:735-739, 141-142). A landing a few ulps short of a whole tick is
    on the earlier tick today and on that tick in the kernel."""
    if not _wall(script):
        return
    for d in p.decisions.values():
        approach = d["approach"]
        if approach is None:
            continue
        whole = round(approach["requested"])
        if not (0 < whole - approach["requested"] < TOLERANCE and approach["from"] == whole - 1):
            continue
        for record, name in (
            (approach, "from"),
            (approach, "to"),
            (d["resolved"], "environment_tick"),
            (d["boundary"], "from"),
        ):
            if record is not None and record[name] == whole - 1:
                record[name] = whole


def _u1(script: Script, p: Projection, notes: dict) -> None:
    """The kernel's Outcome carries the result's output tokens on the wall clock too
    (turns.py:335-340, 627); today's lifecycle writes None there (runner.py:544-547)."""
    if not _wall(script):
        return
    for k, d in p.decisions.items():
        outcome = d["outcome"]
        if outcome is None or outcome["action_output_tokens"] is not None:
            continue
        if p.facts["timeout_reasons"].get(k) is None and k < len(script.turns):
            outcome["action_output_tokens"] = script.turns[k].output_tokens


def _x1(script: Script, p: Projection, notes: dict) -> None:
    """A wait is time the kernel lets pass, not an action the environment executes
    (turns.py:449-450); its record has no active holds or mouse buttons (turns.py:518-525)."""
    for d in p.decisions.values():
        d["executed"] = [e for e in d["executed"] if not e["action"].startswith("wait:")]
        if d["wait_record"] is not None:
            d["wait_record"].pop("active_holds", None)
            d["wait_record"].pop("active_mouse_buttons", None)


def _kernel_end(script: Script, p: Projection) -> tuple[float, int | None] | None:
    """Wall clock: when and in which decision the kernel's periodic check finds the task's end
    before today's loop does, or None when today's loop finds it first."""
    if not _wall(script) or script.ends_at_s is None:
        return None
    end_s = p.episode["termination"]["end_s"]
    found = _grid_check(script, script.ends_at_s)
    if found is None or earlier(end_s, found):
        return None
    k = _in_hand_at(p, found)
    if _same(found, end_s):
        # Both find it at the same instant; that is a periodic check only when the decision is
        # still pacing then: the check due there was set one environment frame earlier, before
        # the landing set the wake (agenda order, kernel/agenda.py). At one instant a landing
        # comes before a periodic check (Priority.COMPLETION before START).
        committed = p.facts["committed_at"].get(k)
        if k is None or committed is None or not _same(committed, found):
            return None
        paced_from = p.facts["resolved_at"].get(k)
        if paced_from is None or not earlier(found - 1.0 / script.env_fps, paced_from):
            return None
    return found, k


def _kb(script: Script, p: Projection, notes: dict) -> None:
    """The wall clock checks the end every 1/env_fps s, during generations, waits and pacing too
    (turns.py:198-200, 659-663); an end found so is dated at the asking. Today only the loop top
    and the checks around landing ask (runner.py:1259, 571-589)."""
    found = _kernel_end(script, p)
    if found is None:
        return
    at, k = found
    termination = p.episode["termination"]
    if termination["stop_reason"] == "think_budget":
        return  # K-i reads it
    notes["ended"] = (at, k)
    invalid = p.episode["invalid"]
    if earlier(at, termination["end_s"]) and (
        invalid is None or invalid["reason"] != "evaluation_snapshot_failed"
    ):
        # What ended today's episode later, the deadline or a failure, never comes.
        termination["stop_reason"] = script.end_reason or "game_decided"
        p.episode["invalid"], p.episode["invalidation_records"] = None, 0
    _end_at(script, p, notes, k, at)


def _end_at(script: Script, p: Projection, notes: dict, k: int | None, at: float) -> None:
    """The episode ends at ``at`` with decision k in hand, or none."""
    p.episode["termination"]["end_s"] = round(at, 9)
    p.episode["key_ups"] = key_releases(p.facts["key_ups"], at)
    notes["in_hand"] = None
    if k is not None:
        _cut(script, p, notes, k, at)


def _cut(script: Script, p: Projection, notes: dict, k: int, at: float) -> None:
    """Settle decision k as the kernel does when the episode ends at ``at`` (turns.py:248-266)."""
    facts, d = p.facts, p.decisions[k]
    _drop_after(p, k)
    if earlier(at, facts["finished_at"].get(k, math.inf)):
        _give_up(p, k)
        notes["in_hand"] = None
        return
    notes["in_hand"] = k
    if earlier(at, facts["resolved_at"].get(k, math.inf)):
        return  # K-c: a wait still running
    boundary = d["boundary"]
    if boundary is not None and earlier(at, facts["committed_at"][k]):
        boundary["to"] = _floor(at * script.env_fps)
        boundary["dropped"] = _dropped(script, d["observed"], boundary["to"])
        d["outcome"]["dropped_intermediate_frames"] = boundary["dropped"]


def _kc(script: Script, p: Projection, notes: dict) -> None:
    """A wall wait is a kernel timer that the end interrupts (turns.py:497-526); today the
    environment sleeps it out, cut only at the deadline (sandbox/actions.py:877-880). On the token
    clock the kernel steps the wait itself, so a step that fails cuts it short like an end
    (turns.py:707-711, 260-261); today the environment turns that failure into a failed action
    (sandbox/actions.py:904-905)."""
    if not _wall(script):
        return _kc_token(script, p)
    ended = notes.get("ended")
    if ended is None or ended[1] is None:
        return
    at, k = ended
    d, facts = p.decisions[k], p.facts
    record = d["wait_record"]
    landed, resolved = facts["finished_at"].get(k), facts["resolved_at"].get(k)
    if record is None or landed is None or earlier(at, landed) or not earlier(at, resolved):
        return
    record["duration_simulated_seconds"] = round(max(0.0, at - landed), 9)
    tick = _floor(at * script.env_fps)
    d["resolved"]["environment_tick"] = tick
    d["resolved"]["terminal_after_action"] = True
    d["outcome"]["terminal_after_action"] = True
    d["outcome"]["next_observation_tick"] = tick


def _kc_token(script: Script, p: Projection) -> None:
    invalid = p.episode["invalid"]
    if invalid is None or invalid["reason"] != "clock_advance_failed":
        return
    for d in p.decisions.values():
        resolved, proposed = d["resolved"], d["proposed"] or ""
        if (
            resolved is None
            or resolved["status"] != "invalid_run"
            or not proposed.startswith("wait:")
        ):
            continue
        start, reached = d["approach"]["to"], resolved["environment_tick"]
        requested = float(proposed.split(":")[1])
        elapsed = round(max(0.0, (reached - start) / script.env_fps), 9)
        horizon_truncates = start + requested * script.env_fps > script.max_ticks + 1e-9
        by_horizon = horizon_truncates and reached >= script.max_ticks - 1e-9
        d["wait_record"] = {
            "duration_simulated_seconds": elapsed,
            "requested_duration_simulated_seconds": requested,
            "truncated_by_episode_horizon": by_horizon,
            "truncated_by_terminal_condition": elapsed + 1e-9 < requested and not by_horizon,
        }
        d["applied"] = True
        for record in (resolved, d["commit"], d["outcome"]):
            record["status"] = "wait"
        resolved["terminal_after_action"] = d["outcome"]["terminal_after_action"] = True


def _ki(script: Script, p: Projection, notes: dict) -> None:
    """When the think budget runs out and the task ends, the first signal names the reason
    (core.py:175-185); today the budget always wins (runner.py:838-843)."""
    termination = p.episode["termination"]
    if termination["stop_reason"] != "think_budget":
        return
    found = _kernel_end(script, p)
    if found is None:
        return
    at, k = found
    notes["ended"] = (at, k)
    termination["stop_reason"] = script.end_reason or "game_decided"
    _end_at(script, p, notes, k, at)


def _ka(script: Script, p: Projection, notes: dict) -> None:
    """The wall deadline is a timer without today's grace (runner.py:353): at the deadline the
    kernel freezes the snapshot and gives up on the generation in flight, whose decision is not
    reported; a result back exactly at the deadline is discarded (turns.py:226-227, core.py)."""
    if not _ended_by_the_deadline(script, p, notes):
        return
    deadline, facts = script.duration_s, p.facts
    for k in sorted(p.decisions):
        finished = facts["finished_at"].get(k)
        if earlier(facts["started_at"].get(k, math.inf), deadline) and (
            finished is None or not earlier(finished, deadline)
        ):
            _give_up(p, k)
            notes["in_hand"] = None


def _ended_by_the_deadline(script: Script, p: Projection, notes: dict) -> bool:
    """Today's wall episode ran into the deadline: nothing else ended it first. A failed
    snapshot read turns its stop reason into invalid_run afterwards (runner.py:832-839)."""
    if not _wall(script) or "ended" in notes:
        return False
    termination, ended_by = p.episode["termination"], p.facts["ended_by"]
    return termination["stop_reason"] != "think_budget" and ended_by in (None, "deadline")


def _kg(script: Script, p: Projection, notes: dict) -> None:
    """The kernel freezes the snapshot at the first end signal, so on the wall clock it is read
    then (core.py:175-183), and the record dates the end at that signal even when the read fails
    (core.py:221-238); today the snapshot is read once the loop has exited, and a failed read
    dates the end itself (runner.py:812-835, 862-869)."""
    termination = p.episode["termination"]
    if not _wall(script):
        return
    if termination["snapshot_s"] is not None:
        termination["snapshot_s"] = round(termination["end_s"] + script.snapshot_s, 9)
        return
    if "ended" in notes:
        return  # K-b dated it
    if _ended_by_the_deadline(script, p, notes):
        termination["end_s"] = round(script.duration_s, 9)
    elif p.facts["ended_at"] is not None:
        termination["end_s"] = round(p.facts["ended_at"], 9)


def _kh(script: Script, p: Projection, notes: dict) -> None:
    """A frozen stream invalidates the run without aborting the agent (turns.py:289-293); today
    the watchdog also calls adapter.abort() (runner.py:286-293)."""
    invalid = p.episode["invalid"]
    if invalid is not None and invalid["reason"] == "observation_stream_frozen":
        p.episode["aborts"] -= 1


UNUSABLE = {
    "missing_output_token_usage",
    "generation_error",
    "generation_watchdog_timeout",
    "inspection_protocol_error",
    "trajectory_write_failed",
}


def _kf(script: Script, p: Projection, notes: dict) -> None:
    """An unusable result still resolves the action it proposed, as invalid_run
    (turns.py:392-406); today _apply returns before it is resolved (runner.py:499-531)."""
    invalid = p.episode["invalid"]
    if invalid is None or invalid["reason"] not in UNUSABLE:
        return
    for d in p.decisions.values():
        if d["proposed"] is not None and d["resolved"] is None and d["commit"] is None:
            d["resolved"] = {
                "status": "invalid_run",
                "terminal_after_action": False,
                "action_clock_position": None,
                "environment_tick": None,
            }


def _kj(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: a lease that ends inside a tap's own 1/60 s is released where the tap ends,
    by the kernel's next step (turns.py:693-705); today the tap's advance splits there
    (sandbox/actions.py:548, 246-253)."""
    if _wall(script):
        return
    taps, seconds = p.facts["taps"], 1.0 / script.env_fps
    released = []
    for logged in p.facts["key_ups"]:
        at = logged["at"]
        tick = at / seconds
        ends = [to for start, to in taps if start - TOLERANCE <= tick <= to + TOLERANCE]
        if logged["in_tap"] and ends:
            at = ends[0] * seconds
        released.append({"at": at, "keys": logged["keys"]})
    p.episode["key_ups"] = key_releases(released, p.episode["termination"]["end_s"])


def _kn(script: Script, p: Projection, notes: dict) -> None:
    """A failed read in the check right before landing settles the decision as invalid_run
    (turns.py:430-441); today the environment's True reads as the task's end and the decision is
    not_applied_terminal (sandbox/actions.py:1004-1010, runner.py:572-573)."""
    invalid = p.episode["invalid"]
    if invalid is None or invalid["reason"] != "terminal_state_check_failed":
        return
    for d in p.decisions.values():
        for record in (d["resolved"], d["commit"], d["outcome"]):
            if record is not None and record["status"] == "not_applied_terminal":
                record["status"] = "invalid_run"


def _ke(script: Script, p: Projection, notes: dict) -> None:
    """An environment fault that escapes today's loop leaves the decision in hand unreported
    (runner.py:1257-1268); the kernel ends the episode on it and reports that decision after the
    snapshot (turns.py:248-266). A failed token step while pacing reports the decision with its
    own status and the boundary it reached, as today's loop would have written them
    (runner.py:644-679)."""
    invalid = p.episode["invalid"]
    if (
        p.facts["loop_error"] is None
        or invalid is None
        or invalid["reason"] != "clock_advance_failed"
    ):
        return
    k = max(p.decisions)
    d = p.decisions[k]
    if d["commit"] is not None:
        return
    # The failed step did not move the clock: it stands where the last check asked.
    reached = p.episode["checks"][-1]
    t_k, tokens = d["observed"], p.facts["output_tokens"].get(k)
    frame_ticks = script.env_fps / script.input_fps
    if d["resolved"] is None:
        # The step to the action time failed: the action is not landed (turns.py:408-428).
        tau = t_k + tokens / script.tokens_per_frame * frame_ticks  # runner.py:535
        d["approach"] = {"from": t_k, "requested": min(tau, script.max_ticks), "to": reached}
        d["resolved"] = {
            "status": "invalid_run",
            "terminal_after_action": False,
            "action_clock_position": round(tau, 9),
            "environment_tick": reached,
        }
        d["commit"] = {"status": "invalid_run", "detail": ""}
        d["outcome"] = {
            "status": "invalid_run",
            "terminal_after_action": False,
            "action_clock_position": round(tau, 9),
            "next_observation_tick": reached,
            "dropped_intermediate_frames": 0,
            "post_action_rendered": False,
            "token_delay_frames": tau - t_k,
            "action_output_tokens": tokens,
            "post_horizon_output_tokens": 0,
            "frame_age_s": round((tau - t_k) / script.env_fps, 6),
        }
        return
    tau, landed = d["resolved"]["action_clock_position"], d["resolved"]["environment_tick"]
    next_frame = max(_floor(t_k / frame_ticks) + 1, _whole_up(max(tau, landed) / frame_ticks))
    next_tick = min(_whole_up(next_frame * frame_ticks), script.max_ticks)
    dropped = _dropped(script, t_k, reached)
    d["boundary"] = {
        "from": landed,
        "requested": next_tick,
        "to": reached,
        "rendered": False,
        "dropped": dropped,
    }
    detail = script.turns[k].rejected if d["resolved"]["status"] == "rejected" else ""
    d["commit"] = {"status": d["resolved"]["status"], "detail": detail}
    d["outcome"] = {
        "status": d["resolved"]["status"],
        "terminal_after_action": False,
        "action_clock_position": tau,
        "next_observation_tick": next_tick,
        "dropped_intermediate_frames": dropped,
        "post_action_rendered": False,
        "token_delay_frames": tau - t_k,
        "action_output_tokens": tokens,
        "post_horizon_output_tokens": 0,
        "frame_age_s": round((tau - t_k) / script.env_fps, 6),
    }


def _whole_up(value: float) -> int:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) < 1e-9 else math.ceil(value)


def _krel(script: Script, p: Projection, notes: dict) -> None:
    """A wall hold's lease starts when the kernel sends Execute (turns.py:451-457); today it
    starts when the environment presses the keys, after its guards (sandbox/actions.py:746)."""
    if not _wall(script) or not script.execute_s:
        return
    for released in p.episode["key_ups"]:
        released["at"] = round(released["at"] - script.execute_s, 9)


def _u2(script: Script, p: Projection, notes: dict) -> None:
    """The kernel writes RunInvalidated for every failure (core.py:166-173), also one the
    environment latched itself; today only the runner's own _invalidate writes it
    (runner.py:200-211)."""
    if p.episode["invalid"] is not None and p.episode["invalidation_records"] == 0:
        p.episode["invalidation_records"] = 1


def _u3(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: once a check finds the end, the kernel asks no more (turns.py:665-677); when
    today's environment finds it inside an advance, the loop top asks again
    (sandbox/actions.py:279-280, runner.py:1259), and that read can fail and void the run."""
    if _wall(script):
        return
    answers = p.facts["check_answers"]
    first = next((i for i, answer in enumerate(answers) if answer), None)
    if first is None:
        return
    del p.episode["checks"][first + 1 :]
    if answers[first] == "task" and "failed" in answers[first + 1 :]:
        p.episode["invalid"], p.episode["invalidation_records"] = None, 0
        p.episode["termination"]["stop_reason"] = script.end_reason or "game_decided"


def _x2(script: Script, p: Projection, notes: dict) -> None:
    """The kernel freezes the snapshot before it reports the decision in hand (core.py:183,
    turns.py:248-266); today the decision is reported first and the snapshot read after the loop
    (runner.py:714, 1268)."""
    k = notes["in_hand"] if "in_hand" in notes else _old_in_hand(script, p, notes)
    if k is not None and k in p.decisions and p.decisions[k]["commit"] is not None:
        p.episode["snapshot_after"] -= 1


def _old_in_hand(script: Script, p: Projection, notes: dict) -> int | None:
    """The decision today's loop committed after the first end signal: an end or a failure found
    before its commit, or the wall deadline passing while it paced or waited."""
    committed = p.facts["committed_at"]
    if not committed:
        return None
    last = max(committed)
    if p.facts["ended_before_commit"].get(last):
        return last
    if _ended_by_the_deadline(script, p, notes) and not earlier(committed[last], script.duration_s):
        return last
    return None


RULES = (
    Rule("X4", "ticks and frame positions agree within 1e-9", _x4),
    Rule("U1", "wall Outcome.action_output_tokens carries the result's tokens", _u1),
    Rule("X1", "a wait is not executed by the environment; its record has no holds", _x1),
    Rule("K-b", "the wall clock checks the end every 1/env_fps s", _kb),
    Rule("K-c", "a wall wait is a timer the end interrupts", _kc),
    Rule("K-i", "think budget and the task's end: the first names the reason", _ki),
    Rule("K-a", "the wall deadline is a timer without grace", _ka),
    Rule(
        "K-m",
        "no empty pacing record after the wall deadline",
        seen="subsumed by K-a: today writes it only for a reply past the deadline",
    ),
    Rule("K-g", "the termination record's snapshot time is the end's", _kg),
    Rule("K-h", "a frozen stream does not abort the agent", _kh),
    Rule("K-f", "an unusable result resolves its action as invalid_run", _kf),
    Rule("K-j", "a lease ending inside a token tap is released where the tap ends", _kj),
    Rule("K-n", "a failed read before landing is invalid_run, not not_applied_terminal", _kn),
    Rule("K-e", "an escaping fault: the decision in hand is settled and reported", _ke),
    Rule("K-rel", "a wall hold's lease starts at Execute", _krel),
    Rule(
        "K-d",
        "nothing advances after the first end signal",
        seen="not found: today's loop never advances past an end it was told of",
    ),
    Rule(
        "K-k",
        "the wall clock does not compare the frame count separately",
        seen="not observable: on the wall the frame count is the deadline's",
    ),
    Rule(
        "K-l",
        "the per-move cap runs from StartGeneration",
        seen="not observable: no time passes between the capture and _run_step here",
    ),
    Rule(
        "K-poll",
        "waiting for the next wall frame is a timer, not a 30 ms poll",
        seen="not observable: FakeEnvironment lands on whole ticks (fake_env.py:113-133)",
    ),
    Rule("U3", "no state read after the token end was found inside an advance", _u3),
    Rule("U2", "an environment-latched failure is written as RunInvalidated", _u2),
    Rule("X2", "the snapshot is frozen before the decision in hand is reported", _x2),
    Rule("X3", "no synthetic InferenceCompleted for a given-up generation", seen="shell only"),
    Rule("X5", "KernelRecord entries in the episode trace", seen="shell only"),
    Rule("X6", "a timed-out step's feedback does not rewrite the previous record", seen="P2"),
    Rule("X7", "DONE turned into a wait is reported as actual=wait", seen="P2"),
    Rule("X8", "generation numbers in the agent trace; Decision client rebuilt", seen="P2"),
    Rule("X9", "agent events numbered by the kernel", seen="P1d"),
    Rule("X10", "SessionReleaseWait agent records", seen="P3, live drivers only"),
    Rule("X11", "token key events and releases bounded by the environment budget", seen="P5b"),
    Rule("X12", "three consecutive failed reads before the run is invalid", seen="W0"),
    Rule("X13", "a periodic wall state read is bounded by 10 s", seen="shell and environment"),
)
ORDER = [rule.id for rule in RULES]
