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
    if old.facts.get("x12_shown") and "X12" not in verdict.triggered:
        verdict.triggered.append("X12")  # it changed what today's loop was made to read
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
    """Input frames a decision crossed but one (runner.py:579-580)."""
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
    j >= 1, while the episode lasts (turns.py:235-240, 671-675)."""
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


# ---- X12: which failed read ends the run ------------------------------------------------------

READS_IN_A_ROW = 3  # turns.py:63 READ_FAILURE_LIMIT


@dataclass(frozen=True)
class X12Plan:
    """What today's loop plays for a script with failed reads, so that it shows what the kernel
    does with them. ``script`` fails only the read the kernel ends the run on, and takes the
    failures before it as reads that saw no end; ``end_at`` is that read's time on the wall
    clock when it is one of the kernel's periodic reads, which today's loop never makes, for the
    rule to place; ``shown`` is whether the kernel ends anywhere else than on today's first
    failed read."""

    script: Script
    end_at: float | None
    shown: bool


def x12_probe(script: Script) -> Script:
    """Today's loop with every scripted failure taken as a read that saw no end: until a third
    failure in a row the kernel reads the episode the same way (turns.py:691-700)."""
    if _wall(script):
        return script.but(failing_reads_s=())
    return script.but(
        check_errors=(),
        tap_check_errors=(),
        blind_checks=script.check_errors,
        blind_tap_checks=script.tap_check_errors,
    )


def x12_plan(script: Script, probe: Projection) -> X12Plan:
    """The kernel counts failed reads in the order it handles them: end checks, and on the token
    clock the reads a key tap makes, before the check after it (turns.py:471-474). A read that
    succeeds resets the count, whatever it answers; the third failure in a row ends the run. Today
    the first failure does (sandbox/actions.py:1004-1010)."""
    if _wall(script):
        return _x12_wall(script, probe)
    failing = {"check": set(script.check_errors), "tap": set(script.tap_check_errors)}
    first = third = None
    in_a_row = 0
    for kind, read, answer in probe.facts["reads"]:
        if read not in failing[kind]:
            if answer:
                break  # the end came before a third failure in a row
            in_a_row = 0
            continue
        first = first or (kind, read)
        in_a_row += 1
        if in_a_row == READS_IN_A_ROW:
            third = (kind, read)
            break
    kept = {} if third is None else {third[0]: (third[1],)}
    played = script.but(
        check_errors=kept.get("check", ()),
        tap_check_errors=kept.get("tap", ()),
        blind_checks=tuple(sorted(failing["check"] - set(kept.get("check", ())))),
        blind_tap_checks=tuple(sorted(failing["tap"] - set(kept.get("tap", ())))),
    )
    return X12Plan(played, None, first is not None and first != third)


def _x12_wall(script: Script, probe: Projection) -> X12Plan:
    """Besides today's reads, the wall kernel reads every 1/env_fps s while the episode lasts
    (turns.py:235-240, 671-675)."""
    fps, end = script.env_fps, probe.episode["termination"]["end_s"]
    reads = [(at, "loop") for at, _ in probe.facts["reads"]]
    j = 1
    while j / fps < script.duration_s and earlier(j / fps, end):
        reads.append((j / fps, "periodic"))
        j += 1
    reads.sort(key=lambda read: read[0])  # stable: today's reads at one instant keep their order
    first = next(
        (at for at, source in reads if source == "loop" and script.read_fails_at(at)), None
    )
    third, in_a_row = None, 0
    for index, (at, _) in enumerate(reads):
        if not script.read_fails_at(at):
            in_a_row = 0
            continue
        in_a_row += 1
        if in_a_row == READS_IN_A_ROW:
            third = index
            break
    if third is None:
        return X12Plan(script.but(failing_reads_s=()), None, first is not None)
    at, source = reads[third]
    # Today's reads at one instant come in a known order, a periodic read among them does not;
    # and a window can only fail the first of today's reads at an instant.
    same = [index for index, (other, _) in enumerate(reads) if _same(other, at)]
    if same[0] != third or len(same) > 1 and any(reads[i][1] == "periodic" for i in same):
        raise ValueError(f"the third failed read in a row is not alone first at {at:g} s")
    if source == "periodic":
        return X12Plan(script.but(failing_reads_s=()), at, True)
    played = script.but(failing_reads_s=((at - TOLERANCE, at + TOLERANCE),))
    return X12Plan(played, None, first is None or not _same(first, at))


# ---- the rules --------------------------------------------------------------------------------


def _x4(script: Script, p: Projection, notes: dict) -> None:
    """Today's wall tick is int(clock position) (sandbox/actions.py:200-203); the kernel floors the
    position plus 1e-9 (turns.py:758-762, 146-147). A landing a few ulps short of a whole tick is
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
    (turns.py:342-346, 637); today's lifecycle writes None there (runner.py:459-462)."""
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
    (turns.py:455-456); its record has no active holds or mouse buttons (turns.py:528-535)."""
    for d in p.decisions.values():
        d["executed"] = [e for e in d["executed"] if not e["action"].startswith("wait:")]
        if d["wait_record"] is not None:
            d["wait_record"].pop("active_holds", None)
            d["wait_record"].pop("active_mouse_buttons", None)


def _x12(script: Script, p: Projection, notes: dict) -> None:
    """The kernel invalidates the run on the third failed state read in a row (turns.py:691-700),
    today's loop on the first; compare() has today's loop fail only that third read
    (x12_plan). On the wall clock it can be a periodic read today's loop never makes: the run
    ends there, invalidated."""
    at = p.facts.get("x12_end")
    if at is None:
        return
    p.episode["termination"]["stop_reason"] = "invalid_run"
    p.episode["invalid"] = {"category": "environment", "reason": "terminal_state_check_failed"}
    p.episode["invalidation_records"] = 1
    k = _in_hand_at(p, at)
    notes["ended"] = (at, k)
    _end_at(script, p, notes, k, at)


def _kernel_end(script: Script, p: Projection, notes: dict) -> tuple[float, int | None] | None:
    """Wall clock: when and in which decision the kernel's periodic check finds the task's end
    before today's loop does, or None when today's loop finds it first."""
    if not _wall(script) or script.ends_at_s is None or "ended" in notes:
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
    (turns.py:205-206, 671-675); an end found so is dated at the asking. Today only the loop top
    and the checks around landing ask (runner.py:858, 485-504)."""
    found = _kernel_end(script, p, notes)
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
    p.episode["captures"] = [c for c in p.episode["captures"] if earlier(c["timestamp"], at)]
    notes["in_hand"] = None
    if k is not None:
        _cut(script, p, notes, k, at)


def _cut(script: Script, p: Projection, notes: dict, k: int, at: float) -> None:
    """Settle decision k as the kernel does when the episode ends at ``at`` (turns.py:254-272)."""
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
    """A wall wait is a kernel timer that the end interrupts (turns.py:507-536); today the
    environment sleeps it out, cut only at the deadline (sandbox/actions.py:877-880). On the token
    clock the kernel steps the wait itself, so a step that fails cuts it short like an end
    (turns.py:730-734, 266-267); today the environment turns that failure into a failed action
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
    (core.py:176-186); today the budget always wins (runner.py:700-705)."""
    termination = p.episode["termination"]
    if termination["stop_reason"] != "think_budget":
        return
    if not _wall(script):
        if p.facts["ended_by"] == "task":
            termination["stop_reason"] = script.end_reason or "game_decided"
        return
    found = _kernel_end(script, p, notes)
    if found is None:
        return
    at, k = found
    notes["ended"] = (at, k)
    termination["stop_reason"] = script.end_reason or "game_decided"
    _end_at(script, p, notes, k, at)


def _kd(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: nothing advances after the first end signal (turns.py:556-562, 255). Today a
    decision skipped before landing (no action, over the budget) skips the check before landing,
    so after an end found while moving to its action time it still paces to the next frame
    (runner.py:479-484, 554-563), and the end is dated where that pacing stopped."""
    ended_at = p.facts["ended_at"]
    if _wall(script) or ended_at is None:
        return
    seconds = 1.0 / script.env_fps
    for k, d in p.decisions.items():
        boundary = d["boundary"]
        if (
            boundary is None
            or not p.facts["ended_before_commit"].get(k)
            or earlier(boundary["from"] * seconds, ended_at)
        ):
            continue
        d["boundary"] = None
        d["outcome"]["next_observation_tick"] = boundary["from"]
        d["outcome"]["dropped_intermediate_frames"] = 0
        termination = p.episode["termination"]
        termination["end_s"] = round(ended_at, 9)
        if termination["snapshot_s"] is not None:
            termination["snapshot_s"] = round(ended_at, 9)


def _ka(script: Script, p: Projection, notes: dict) -> None:
    """The wall deadline is a timer without today's grace (runner.py:294): at the deadline the
    kernel freezes the snapshot and gives up on the generation in flight, whose decision is not
    reported; a result back exactly at the deadline is discarded (turns.py:233-234, core.py)."""
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
    snapshot read turns its stop reason into invalid_run afterwards (runner.py:694-701)."""
    if not _wall(script) or "ended" in notes:
        return False
    termination, ended_by = p.episode["termination"], p.facts["ended_by"]
    return termination["stop_reason"] != "think_budget" and ended_by in (None, "deadline")


def _kg(script: Script, p: Projection, notes: dict) -> None:
    """The kernel freezes the snapshot at the first end signal, so on the wall clock it is read
    then (core.py:176-184), and the record dates the end at that signal even when the read fails
    (core.py:222-239); today the snapshot is read once the loop has exited, and a failed read
    dates the end itself (runner.py:674-697, 724-735)."""
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
    """A frozen stream invalidates the run without aborting the agent (turns.py:295-299); today
    the watchdog also calls adapter.abort() (runner.py:226-233)."""
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
    (turns.py:398-412); today _apply returns before it is resolved (runner.py:433-446)."""
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
    by the kernel's next step (turns.py:716-728); today the tap's advance splits there
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
    (turns.py:436-447); today the environment's True reads as the task's end and the decision is
    not_applied_terminal (sandbox/actions.py:1004-1010, runner.py:487-488)."""
    invalid = p.episode["invalid"]
    if invalid is None or invalid["reason"] != "terminal_state_check_failed":
        return
    for d in p.decisions.values():
        for record in (d["resolved"], d["commit"], d["outcome"]):
            if record is not None and record["status"] == "not_applied_terminal":
                record["status"] = "invalid_run"


def _ke(script: Script, p: Projection, notes: dict) -> None:
    """An environment fault that escapes today's loop leaves the decision in hand unreported
    (runner.py:857-867); the kernel ends the episode on it and reports that decision after the
    snapshot (turns.py:254-272). A failed token step while pacing reports the decision with its
    own status and the boundary it reached, as today's loop would have written them
    (runner.py:554-594)."""
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
        # The step to the action time failed: the action is not landed (turns.py:414-434).
        tau = t_k + tokens / script.tokens_per_frame * frame_ticks  # runner.py:450
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
    """A wall hold's lease starts when the kernel sends Execute (turns.py:457-463); today it
    starts when the environment presses the keys, after its guards (sandbox/actions.py:746)."""
    if not _wall(script) or not script.execute_s:
        return
    for released in p.episode["key_ups"]:
        released["at"] = round(released["at"] - script.execute_s, 9)


def _u2(script: Script, p: Projection, notes: dict) -> None:
    """The kernel writes RunInvalidated for every failure (core.py:166-174), also one the
    environment latched itself; today only the runner's own _invalidate writes it
    (runner.py:197-208)."""
    if p.episode["invalid"] is not None and p.episode["invalidation_records"] == 0:
        p.episode["invalidation_records"] = 1


def _u3(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: once a read finds the end, or fails for the last time, the kernel asks no more
    (turns.py:677-689, 471-474); when today's environment finds it inside an advance or a tap,
    the loop asks again after execution or at its top (sandbox/actions.py:279-280,
    runner.py:504, 858). That extra read can also fail and void the run today; the kernel never
    makes it, so X12's plan never fails it."""
    if _wall(script):
        return
    asked = 0
    for kind, _, answer in p.facts["reads"]:
        asked += kind == "check"
        if answer:
            del p.episode["checks"][asked:]
            return


def _x2(script: Script, p: Projection, notes: dict) -> None:
    """The kernel freezes the snapshot before it reports the decision in hand (core.py:184,
    turns.py:254-272); today the decision is reported first and the snapshot read after the loop
    (runner.py:629, 867)."""
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
    Rule("X12", "the third failed state read in a row invalidates the run", _x12),
    Rule("K-b", "the wall clock checks the end every 1/env_fps s", _kb),
    Rule("K-c", "a wall wait is a timer the end interrupts", _kc),
    Rule("K-i", "think budget and the task's end: the first names the reason", _ki),
    Rule("K-d", "nothing advances after the first end signal", _kd),
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
    Rule("X13", "a periodic wall state read is bounded by 10 s", seen="shell and environment"),
)
ORDER = [rule.id for rule in RULES]
