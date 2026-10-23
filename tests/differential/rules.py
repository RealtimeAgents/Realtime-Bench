"""The expected differences between today's loop and the kernel, as numbered rewrite rules
(wave2-plan.md, 5 预期差异清单; wave2-overrides.md, X12 and X13).

Each rule rewrites the old side's projection into what the new side shows; the rules run in
order and what is left must match exactly. A rule triggers when it changed something. A script's
``expect`` names the rules it may trigger: one it did not name fails the comparison, one it named
that did not trigger is reported. Rules without a rewrite cannot show between today's loop and the
simulation; ``seen`` says where they can.

U-numbered rules are differences found after the plan's list: U2 by this framework, put on the
list by N5 (with U3 and U6, which KF5 and p1b's fix of L3 took up); U7 to U13 by p1b, U10 and U11
recorded by N10 and N34; U14 by this framework, gone since N44 dated a failed read's end as an end
a read found (K-g). KF-numbered ones are kernel-fix's, D47 is p1b's name for decision 47; p1c
numbers decision 31, KF5 and KF5d here, X18, and the token clock's ticks turned into equal floats
X17.

The old side's times come from its facts: when each generation started and returned, when each
decision was resolved, executed and committed, and every end check with its answer. The random
scripts' operations take no time, but for reading the evaluation snapshot. Named scripts give
captures, state reads and actions their time (scenarios.TIMED, ACROSS; N29), and the rules for it
(K-a, K-b, K-k, D47, K-g and X2) hold where a read is shorter than an environment frame, no reply
comes back at the instant a read is asked, and each decision paces to its next frame: what
today's extra reads delay is not carried into the next decision. A deadline or a periodic read
that falls due during one of the shell's operations is handled once it has returned.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from realtime_framework.harnesses.default.prompts import (
    UNANSWERED_STEP_NOTES,
    UNSUPPORTED_TERMINAL_ACTION_FEEDBACK,
)
from realtime_framework.sandbox.actions import TAP_HOLD_S

from .projection import TOLERANCE, Projection, differences, earlier, ends_the_run, key_releases
from .script import Script, parse


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
    apart: list[str] = field(default_factory=list)  # between the shell and the simulation

    @property
    def ok(self) -> bool:
        return not self.problems and not self.unexpected and not self.apart


def check(script: Script, old: Projection, new: Projection) -> Verdict:
    """Today's loop, ``old``, against the shell, ``new``: the projections once the rules have
    rewritten the old one, then what the projection leaves out (_beyond)."""
    current = old.copy()
    notes: dict = {}
    verdict = Verdict(script)
    for rule in RULES:
        if rule.rewrite is None:
            continue
        before = copy.deepcopy((current.compared, current.beyond))
        rule.rewrite(script, current, notes)
        if (current.compared, current.beyond) != before and rule.id not in verdict.triggered:
            verdict.triggered.append(rule.id)
    verdict.problems, verdict.close = differences(current.compared, new.compared)
    verdict.problems += _beyond(script, current, new)
    if verdict.close and "X4" not in verdict.triggered:
        verdict.triggered.append("X4")
    if old.facts.get("x12_shown") and "X12" not in verdict.triggered:
        verdict.triggered.append("X12")  # it changed what today's loop was made to read
    verdict.triggered.sort(key=ORDER.index)
    verdict.unexpected = [rule for rule in verdict.triggered if rule not in script.expect]
    verdict.unused = sorted(set(script.expect) - set(verdict.triggered), key=ORDER.index)
    return verdict


def _beyond(script: Script, old: Projection, new: Projection) -> list[str]:
    """The whole requests and the usage (wave2-plan.md, P4 投影), over the decisions both sides
    started, and the trace records the projection leaves out, decision by decision. The
    decisions today's loop started after the kernel's end, which the rules left out, go with
    their replies and their requests: on the wall clock every request sent once the kernel's
    episode is over, a retry of the generation it gave up on too (K-a); on the token clock, where
    time stands still while a generation runs, the last ones, those decisions' attempts."""
    started = _started(old) & _started(new)
    gone = sorted(set(old.facts["started_at"]) - _started(old))
    end = old.episode["termination"]["end_s"]
    old = old.copy()
    for k in gone:
        _drop_replies(old, k)
    sent = old.beyond["requests"]
    if _wall(script):
        sent = [
            r for r, (_, at) in zip(sent, old.facts["requested"], strict=True) if earlier(at, end)
        ]
    else:
        sent = sent[: len(sent) - sum(len(script.turns[k].attempts) for k in gone)]
    problems = []
    requests = sent, new.beyond["requests"]
    for index, (a, b) in enumerate(zip(*requests)):
        problems += differences(a, b, f"requests[{index}]")[0]
    if _started(old) == _started(new):
        if len(requests[0]) != len(requests[1]):
            problems.append(f"requests: {len(requests[0])} -> {len(requests[1])}")
        problems += differences(old.beyond["usage"], new.beyond["usage"], "usage")[0]
    synthesized = old.beyond["synthesized"]
    for kind, by_decision in old.beyond["trace"].items():
        other = new.beyond["trace"][kind]
        for k in sorted(started - set(synthesized)):
            records = other.get(k, [])
            problems += differences(by_decision.get(k, []), records, f"trace.{kind}[{k}]")[0]
    for k in sorted(started & set(synthesized)):
        expected = synthesized[k]
        streamed = new.beyond["trace"]["OutputChunkReceived"].get(k, [])
        completed = [
            {name: _given_up_field(record, name) for name in expected}
            for record in new.beyond["trace"]["InferenceCompleted"].get(k, [])
        ]
        problems += differences([expected], completed, f"trace.InferenceCompleted[{k}]")[0]
        problems += differences([], streamed, f"trace.OutputChunkReceived[{k}]")[0]
    for k in sorted(started - set(synthesized)):
        a, b = old.beyond["results"].get(k), new.beyond["results"].get(k)
        problems += differences(a, b, f"results[{k}]")[0]
    if old.beyond["key_events"] is not None:
        events = [
            [event for event in side.beyond["key_events"] if not earlier(end, event["at_s"])]
            for side in (old, new)
        ]
        problems += differences(*events, "key_events")[0]
    # A frame captured for a decision that never started is written or not; what is written of
    # it is a frame's record, as for the others.
    frames = old.beyond["trace"]["ObservationCaptured"], new.beyond["trace"]["ObservationCaptured"]
    for k in range(len(old.episode["captures"])):
        a, b = (len(by_decision.get(k, [])) for by_decision in frames)
        if k not in started and a != b:
            problems.append(f"trace.ObservationCaptured[{k}]: {a} record(s) -> {b}")
    return problems


def _given_up_field(record: dict, name: str):
    if name == "provider_usage_missing_reason":
        return (record.get("token_accounting") or {}).get(name)
    return record.get(name)


def _started(p: Projection) -> set[int]:
    return {k for k, d in p.compared["decisions"].items() if d["observed"] is not None}


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
    """Input frames a decision crossed but one (runner.py _apply)."""
    crossed = max(0, _floor(_frames(script, reached)) - _floor(_frames(script, t_k)))
    return max(0, crossed - 1)


# Why the kernel gave a generation up (_give_up), and the reason the shell's record of the call
# gives for its missing usage (evaluation/shell.py _Shell._abandoned_on_a_bound).
GIVEN_UP = {
    "deadline": "generation_aborted_at_wall_timeout",
    "task_end": "generation_aborted_at_task_end",
    "invalid_run": "generation_aborted_at_invalid_run",
}


def _give_up(p: Projection, notes: dict, k: int, reason: str) -> None:
    """Decision k's generation was given up on, for one of GIVEN_UP's reasons: it keeps its
    observation and nothing else, and the agent is aborted, unless today's loop aborted it too
    (its bounded wait ran out). Its reply records are the rules' that name the reason
    (_abandoned)."""
    notes.setdefault("given_up", {})[k] = reason
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
    p.beyond["results"].pop(k, None)


def _drop_after(p: Projection, k: int) -> None:
    for later in [j for j in p.decisions if j > k]:
        if p.decisions[later]["commit"] is not None:
            p.episode["snapshot_after"] -= 1
        del p.decisions[later]


def _grid_check(script: Script, p: Projection, at: float) -> tuple[float, bool] | None:
    """The first periodic wall end check at or after ``at``: the kernel asks at j / env_fps,
    j >= 1, while the episode lasts (turns.py TurnPolicy.on_timer, _schedule_end_check), and
    whether an action executing then put it off. One put off is owed to the read after the
    action, or for a refused action made in its place, both once the execution has returned
    (TurnPolicy.on_timer, _executed)."""
    fps = script.env_fps
    j = max(1, math.ceil(at * fps))
    while j > 1 and (j - 1) / fps >= at:
        j -= 1
    while j / fps < at:
        j += 1
    due = j / fps
    if due >= script.duration_s:
        return None
    for d in p.decisions.values():
        for executed in d["executed"]:
            returned = executed["at"] + script.execute_s
            if earlier(executed["at"], due) and earlier(due, returned):
                return (returned, True) if earlier(returned, script.duration_s) else None
    return due, False


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

READS_IN_A_ROW = 3  # turns.py READ_FAILURE_LIMIT


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
    failure in a row the kernel reads the episode the same way (turns.py TurnPolicy._count_read)."""
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
    clock the reads a key tap makes, before the check after it (turns.py TurnPolicy._executed). A
    read that succeeds resets the count, whatever it answers; the third failure in a row ends the
    run. Today the first failure does (sandbox/actions.py GuestEnvironment.done)."""
    if _wall(script):
        return _x12_wall(script, probe)
    failing = {"check": set(script.check_errors), "tap": set(script.tap_check_errors)}
    first = third = None
    in_a_row = 0
    for kind, read, answer in probe.facts["reads"]:
        if read not in failing[kind]:
            if ends_the_run(answer):
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
    """The wall kernel reads at the start, every 1/env_fps s while the episode lasts and after
    every executed action (turns.py TurnPolicy.on_timer, _schedule_end_check, _check_end); of
    today's reads, those it makes too stand for the reads at their instants, and only they fail
    (WallEnvironment._kernels)."""
    fps, end = script.env_fps, probe.episode["termination"]["end_s"]
    reads = [(at, "loop") for at, _, kernels in probe.facts["reads"] if kernels]
    j = 1
    while j / fps < script.duration_s and earlier(j / fps, end):
        if not any(_same(at, j / fps) for at, _ in reads):
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
    """Today's wall tick is int(clock position) (sandbox/actions.py GuestEnvironment.tick_index);
    the kernel floors the position plus 1e-9 (turns.py _floor, TurnPolicy._tick). A landing a few
    ulps short of a whole tick is on the earlier tick today and on that tick in the kernel."""
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


def _kf2(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: a horizon within 1e-9 above a whole tick is that tick to the kernel
    (kernel/turns.py TurnConfig); today's loop runs to the float main() computes (runner.py main,
    max_minutes * 60 * env_fps). There an action landing on that tick is applied and checked
    after, the loop captures once more, and the episode completes instead of reaching the tick
    limit. kernel-fix 第一轮 第 13 条 made the kernel snap as the environment's clock does and its
    第 5 节 lists this as KF2; today's loop was left as it was."""
    horizon = script.duration_s / 60 * 60 * script.env_fps
    whole = round(horizon)
    if _wall(script) or not 0 < horizon - whole <= 1e-9:
        return
    for k, d in p.decisions.items():
        resolved = d["resolved"]
        if resolved is None or resolved["status"] != "applied":
            continue
        if earlier(resolved["environment_tick"], whole):
            continue
        for record in (resolved, d["commit"], d["outcome"]):
            record["status"] = "not_applied_tick_limit"
        d["applied"], d["executed"], d["boundary"] = False, [], None
        p.beyond["results"].pop(k, None)
        # The check after the execution, one the kernel would make, the last on that tick.
        checks, extras = p.episode["checks"], list(p.facts["check_extras"])
        at = max(i for i, tick in enumerate(checks) if tick == whole and extras[i] is None)
        del checks[at], extras[at]
        p.facts = {**p.facts, "check_extras": extras}
    p.episode["captures"] = [c for c in p.episode["captures"] if earlier(c["tick"], whole)]
    termination = p.episode["termination"]
    if termination["stop_reason"] == "completed":
        termination["stop_reason"] = "tick_limit"


def _kf5(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: the kernel reads the state at most once a frame, the clock's position, and
    once more after every executed action (decision 31; turns.py TurnPolicy._check_end), and
    asks nothing once the episode is over. Today's loop reads again at the loop top and before
    landing, on a tick an advance step just read (KF5a, KF5b), and at the top after the
    environment found the end inside an advance (U3, now KF5a; N5). p1c numbers decision 31
    X18."""
    _cut_checks(script, p, ("frame", "ended"))


def _kf4(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: a tap's read that saw the end ends the episode where the tap stopped, by the
    task's reason (turns.py TurnPolicy._executed); today the loop asks once more after the
    action (runner.py _land_action)."""
    _cut_checks(script, p, ("tap_end",))


def _cut_checks(script: Script, p: Projection, extras: tuple[str, ...]) -> None:
    if _wall(script):
        return
    checks = p.episode["checks"]
    kept = [
        (tick, extra)
        for tick, extra in zip(checks, p.facts["check_extras"], strict=True)
        if extra not in extras
    ]
    p.episode["checks"] = [tick for tick, _ in kept]
    p.facts = {**p.facts, "check_extras": [extra for _, extra in kept]}


def _x1(script: Script, p: Projection, notes: dict) -> None:
    """A wait is time the kernel lets pass, not an action the environment executes
    (turns.py TurnPolicy._pre_checked); its record has no active holds or mouse buttons
    (turns.py TurnPolicy._waited)."""
    for d in p.decisions.values():
        d["executed"] = [e for e in d["executed"] if not e["action"].startswith("wait:")]
        if d["wait_record"] is not None:
            d["wait_record"].pop("active_holds", None)
            d["wait_record"].pop("active_mouse_buttons", None)


def _x12(script: Script, p: Projection, notes: dict) -> None:
    """The kernel invalidates the run on the third failed state read in a row
    (turns.py TurnPolicy._count_read), today's loop on the first; compare() has today's loop fail
    only that third read (x12_plan). On the wall clock it can be a periodic read today's loop never
    makes: the run ends there, invalidated."""
    if not _wall(script):
        return _x12_tap(script, p, notes)
    at = p.facts.get("x12_end")
    if at is None:
        return
    p.episode["termination"]["stop_reason"] = "invalid_run"
    p.episode["invalid"] = {"category": "environment", "reason": "terminal_state_check_failed"}
    p.episode["invalidation_records"] = 1
    k = _in_hand_at(p, at)
    notes["ended"], notes["read_end"] = (at, k), True
    _end_at(script, p, notes, k, at, "invalid_run")


def _x12_tap(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: a failed read does not stop the kernel's tap; it goes on to its end, reading
    after each step, until a read sees the end (W0's ShellEnvironment.kernel_counts_reads,
    simulation.py Simulation._tap), and the run ends there. Today's tap stops at the failure
    (sandbox/actions.py GuestEnvironment.wait_until_tick, _ended_after_step, done)."""
    failed = [n for kind, n, answer in p.facts["reads"] if kind == "tap" and answer == "failed"]
    if not failed:
        return
    start, tick = p.facts["taps"][-1]
    target = min(start + TAP_HOLD_S * script.env_fps, script.max_ticks)
    ends_at = None if script.ends_at_s is None else script.ends_at_s * script.env_fps
    read = failed[-1]
    while tick + 1e-9 < target:
        tick += min(1.0, target - tick)
        tick = round(tick) if abs(tick - round(tick)) < 1e-9 else tick
        read += 1
        if read not in script.tap_check_errors and ends_at is not None and tick >= ends_at:
            break
    notes["tap_end"] = tick
    # Its key goes up where it ends, not where today's stopped.
    stopped = p.facts["taps"][-1][1] / script.env_fps
    for event in p.beyond["key_events"]:
        if event.get("call") == "task.keyUp" and abs(event["at_s"] - stopped) <= 1e-6:
            event["at_s"] = round(tick / script.env_fps, 9)
    k = max(k for k, d in p.decisions.items() if d["executed"])
    p.decisions[k]["resolved"]["environment_tick"] = tick
    p.decisions[k]["outcome"]["next_observation_tick"] = tick
    termination = p.episode["termination"]
    termination["end_s"] = round(tick / script.env_fps, 9)
    if termination["snapshot_s"] is not None:
        termination["snapshot_s"] = termination["end_s"]


def _kernel_end(script: Script, p: Projection, notes: dict) -> tuple[float, int | None] | None:
    """Wall clock: when and in which decision the kernel's periodic check finds the task's end
    before today's loop does, or None when today's loop finds it first."""
    if not _wall(script) or script.ends_at_s is None or "ended" in notes:
        return None
    end_s = p.episode["termination"]["end_s"]
    check = _grid_check(script, p, script.ends_at_s)
    if check is None or earlier(end_s, check[0]):
        return None
    found, put_off = check
    k = _in_hand_at(p, found)
    if _same(found, end_s):
        if put_off:
            # Today's loop reads there too: after the action, or for a refused one at the next
            # loop top, when its pacing has nothing left to wait.
            return None
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
    (turns.py TurnPolicy.start, _schedule_end_check); an end found so is dated at the asking. Today
    only the loop top and the checks around landing ask (runner.py run_stream, _land_action)."""
    found = _kernel_end(script, p, notes)
    if found is None:
        return
    at, k = found
    termination = p.episode["termination"]
    if termination["stop_reason"] == "think_budget":
        return  # K-i reads it
    notes["ended"], notes["read_end"] = (at, k), True
    invalid = p.episode["invalid"]
    if earlier(at, termination["end_s"]) and (
        invalid is None or invalid["reason"] != "evaluation_snapshot_failed"
    ):
        # What ended today's episode later, the deadline or a failure, never comes.
        termination["stop_reason"] = script.end_reason or "game_decided"
        p.episode["invalid"], p.episode["invalidation_records"] = None, 0
    _end_at(script, p, notes, k, at)


def _end_at(
    script: Script, p: Projection, notes: dict, k: int | None, at: float, reason: str = "task_end"
) -> None:
    """The episode ends at ``at`` with decision k in hand, or none; ``reason`` is GIVEN_UP's,
    for the generation it gives up on."""
    p.episode["termination"]["end_s"] = round(at, 9)
    p.episode["key_ups"] = key_releases(p.facts["key_ups"], at)
    p.episode["captures"] = [c for c in p.episode["captures"] if earlier(c["timestamp"], at)]
    notes["in_hand"] = None
    if k is not None:
        _cut(script, p, notes, k, at, reason)


def _cut(script: Script, p: Projection, notes: dict, k: int, at: float, reason: str) -> None:
    """Settle decision k as the kernel does when the episode ends at ``at``
    (turns.py TurnPolicy.on_end)."""
    facts, d = p.facts, p.decisions[k]
    _drop_after(p, k)
    if earlier(at, facts["finished_at"].get(k, math.inf)):
        _give_up(p, notes, k, reason)
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
    """A wall wait is a kernel timer that the end interrupts, and the record says the terminal
    condition cut it (turns.py TurnPolicy._waited); today the environment sleeps it out, cut only at
    the deadline, which neither side flags (legacy_environment.py LegacyWallClock.execute)."""
    if not _wall(script):
        return
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
    record["truncated_by_terminal_condition"] = True
    tick = _floor(at * script.env_fps)
    d["resolved"]["environment_tick"] = tick
    d["resolved"]["terminal_after_action"] = True
    d["outcome"]["terminal_after_action"] = True
    d["outcome"]["next_observation_tick"] = tick


def _ki(script: Script, p: Projection, notes: dict) -> None:
    """When the think budget runs out and the task ends, or on the wall clock the deadline
    passes, the first signal names the reason (core.py Kernel.end); today the budget always wins
    (legacy_loop.py _finalize). Both date the budget's end where the decision over it is reported
    (TurnPolicy._committed; legacy_loop.py _decide), after its pacing; the deadline that pacing
    ran into comes first, with the decision in hand (X2)."""
    termination = p.episode["termination"]
    if termination["stop_reason"] != "think_budget":
        return
    if not _wall(script):
        if p.facts["ended_by"] == "task":
            termination["stop_reason"] = script.end_reason or "game_decided"
        return
    found = _kernel_end(script, p, notes)
    if found is None:
        if not earlier(termination["end_s"], script.duration_s):
            termination["stop_reason"] = "time_limit"
            termination["end_s"] = round(script.duration_s, 9)
        return
    at, k = found
    notes["ended"], notes["read_end"] = (at, k), True
    termination["stop_reason"] = script.end_reason or "game_decided"
    _end_at(script, p, notes, k, at)


def _kd(script: Script, p: Projection, notes: dict) -> None:
    """Token clock: nothing advances after the first end signal (turns.py TurnPolicy._resolve,
    on_end). Today a decision skipped before landing (no action, over the budget) skips the check
    before landing, so after an end found while moving to its action time it still paces to the
    next frame (runner.py _land_action, _apply), and the end is dated where that pacing stopped."""
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
    """The wall deadline is a timer without today's grace (runner.py _move_timeout): at the
    deadline the kernel freezes the snapshot and gives up on the generation in flight, whose
    decision is not reported; a result back exactly at the deadline is discarded
    (turns.py TurnPolicy.on_timer, core.py). The shell records that call as one given up at the
    deadline (decision 48), as today's loop records a bounded wait it gave up; where today's
    loop waited out its grace for the reply instead, that reply's records are gone
    (_abandoned). A frame whose capture returned at the deadline or after starts no generation:
    the kernel handles the deadline first (_deadline_handled); today's loop starts one."""
    if not _ended_by_the_deadline(script, p, notes):
        return
    deadline, facts = script.duration_s, p.facts
    late = [k for k in p.decisions if not earlier(facts["started_at"].get(k, math.inf), deadline)]
    if late:
        _drop_after(p, min(late) - 1)
    for k in sorted(p.decisions):
        finished = facts["finished_at"].get(k)
        if earlier(facts["started_at"].get(k, math.inf), deadline) and (
            finished is None or not earlier(finished, deadline)
        ):
            _give_up(p, notes, k, "deadline")
            notes["in_hand"] = None
    _abandoned(p, notes, "deadline")


def _d47(script: Script, p: Projection, notes: dict) -> None:
    """Wall clock, reads taking time: when the deadline falls while the shell reads the task, a
    result that came back before it, during that read, is filed after it (decision 47; p1b 第四轮
    D47; evaluation/shell.py _Shell._settle_returned): the kernel has given its generation up, the
    decision is not reported, and the result's records are the call's own, its frame usage none,
    since the agent counts the usage of accepted results only (decision 37). The deadline itself
    is handled once the read has returned (K-g). Today's loop takes the result as it comes."""
    if not _ended_by_the_deadline(script, p, notes):
        return
    deadline = script.duration_s
    asked = _read_under_way(script, deadline)
    if asked is None:
        return
    for k in sorted(p.decisions):
        back = p.facts["finished_at"].get(k)
        if back is None or earlier(back, asked) or not earlier(back, deadline):
            continue
        _give_up(p, notes, k, "deadline")
        notes["in_hand"] = None
        for record in p.beyond["trace"]["InferenceCompleted"].get(k, []):
            for name, value in (record["frame_usage"] or {}).items():
                p.beyond["usage"][name] -= value
                record["frame_usage"][name] = 0


def _abandoned(p: Projection, notes: dict, reason: str) -> None:
    """The reply records of the generations given up on for ``reason``: the shell records each
    as a call given up on, with no usage and nothing streamed (decision 48, N10;
    evaluation/shell.py _Shell._abandoned_on_a_bound), and counts no usage of it (decision 37;
    harnesses/default/harness.py DefaultHarness.accept). Today's loop records a reply that came
    back as it came, and a bounded wait it gave up on as the shell does, which stays as it is."""
    trace = p.beyond["trace"]
    for k, why in sorted(notes.get("given_up", {}).items()):
        completed = trace["InferenceCompleted"].get(k, [])
        if why != reason or (why == "deadline" and [r["timed_out"] for r in completed] == [True]):
            continue
        started = trace["InferenceStarted"][k][0]
        usage = completed[0]["frame_usage"] if completed else None
        _drop_replies(p, k)
        p.beyond["synthesized"][k] = abandoned_call(started, reason, usage or {})


def abandoned_call(started: dict, reason: str, usage: dict) -> dict:
    """What the record of a call given up on for ``reason`` says (_abandoned), of the call
    InferenceStarted ``started`` began; ``usage`` is a frame usage, for its fields."""
    return {
        "decision": started["decision"],
        "observation_id": started["observation_id"],
        "timed_out": reason == "deadline",
        "stale": reason != "deadline",
        "error": False,
        "usage": None,
        "provider_usage_missing_reason": GIVEN_UP[reason],
        # The episode's end names itself; the deadline's record is the bound's (N34).
        "harness_abort_reason": None if reason == "deadline" else reason,
        "frame_usage": dict.fromkeys(usage, 0),
    }


def _u10(script: Script, p: Projection, notes: dict) -> None:
    """Wall clock: a generation the task's end gave up on, found by the kernel's periodic check
    (K-b) or with the think budget (K-i), is recorded as a call given up on, marked with its
    reason and left out of the token totals (p1b 第五轮 U10; N10, N34); today the reply comes back
    and is recorded as it came."""
    _abandoned(p, notes, "task_end")


def _u11(script: Script, p: Projection, notes: dict) -> None:
    """Wall clock: a generation an invalidation gave up on (X12's periodic read) is recorded as a
    call given up on (p1b 第五轮 U11; N10); today the reply comes back and is recorded as it
    came."""
    _abandoned(p, notes, "invalid_run")


def _ended_by_the_deadline(script: Script, p: Projection, notes: dict) -> bool:
    """Today's wall episode ran into the deadline: nothing else ended it first. A failed
    snapshot read turns its stop reason into invalid_run afterwards (runner.py _finalize)."""
    if not _wall(script) or "ended" in notes:
        return False
    termination, ended_by = p.episode["termination"], p.facts["ended_by"]
    return termination["stop_reason"] != "think_budget" and ended_by in (None, "deadline")


def _read_ended(p: Projection) -> bool:
    """Today's run ended on a state read: one that saw the task end, or one that failed."""
    invalid = p.episode["invalid"]
    return p.facts["ended_by"] == "task" or (
        p.facts["ended_by"] in ("failed", "invalid_run")
        and invalid is not None
        and invalid["reason"] == "terminal_state_check_failed"
    )


def _kg(script: Script, p: Projection, notes: dict) -> None:
    """The kernel freezes the snapshot at the first end signal, so on the wall clock it is read
    then (core.py Kernel.end), and the record dates the end at that signal even when the read fails
    (core.py Kernel._finish); today the snapshot is read once the loop has exited, and a failed
    read dates the end itself (runner.py _finalize, _termination). The signal is handled, and the
    snapshot read, once the shell is free: an end a state read found, or the failed read that
    invalidates the run, is dated when it was asked and handled when the read has returned
    (turns.py TurnPolicy._end_checked, _count_read; N44), and the deadline once the operation under
    way at it has returned (_deadline_handled)."""
    termination = p.episode["termination"]
    if not _wall(script):
        return
    if termination["snapshot_s"] is not None:
        if _ended_by_the_deadline(script, p, notes):
            handled = _deadline_handled(script, p)
        else:
            read_end = notes.get("read_end", "ended" not in notes and _read_ended(p))
            handled = termination["end_s"] + (script.check_s if read_end else 0.0)
        termination["snapshot_s"] = round(handled + script.snapshot_s, 9)
        return
    if "ended" in notes:
        return  # K-b dated it
    if _ended_by_the_deadline(script, p, notes):
        termination["end_s"] = round(script.duration_s, 9)
    elif p.facts["ended_at"] is not None:
        termination["end_s"] = round(p.facts["ended_at"], 9)


def _kh(script: Script, p: Projection, notes: dict) -> None:
    """A frozen stream invalidates the run without aborting the agent
    (turns.py TurnPolicy._captured); today the watchdog also calls adapter.abort()
    (runner.py _accept_observation)."""
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
    (turns.py TurnPolicy._unusable); today _apply returns before it is resolved
    (runner.py _apply)."""
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
    by the kernel's next step (turns.py TurnPolicy._advance); today the tap's advance splits there
    (sandbox/actions.py GuestEnvironment._execute_clock_key, wait_until_tick)."""
    if _wall(script):
        return
    taps, seconds = p.facts["taps"], 1.0 / script.env_fps
    if "tap_end" in notes:  # X12 ran the last tap on
        taps = [*taps[:-1], (taps[-1][0], notes["tap_end"])]
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
    """Wall clock: a failed read right before landing settles the decision as invalid_run
    (turns.py TurnPolicy._count_read, _resolve), now that the kernel reads there again (N26);
    today the environment's True reads as the task's end and the decision is
    not_applied_terminal (sandbox/actions.py GuestEnvironment.done; legacy_loop.py
    _land_action). The token clock's kernel reads on no tick read already (KF5b)."""
    invalid = p.episode["invalid"]
    if not _wall(script) or invalid is None or invalid["reason"] != "terminal_state_check_failed":
        return
    for d in p.decisions.values():
        for record in (d["resolved"], d["commit"], d["outcome"]):
            if record is not None and record["status"] == "not_applied_terminal":
                record["status"] = "invalid_run"


def _ke(script: Script, p: Projection, notes: dict) -> None:
    """An environment fault that escapes today's loop leaves the decision in hand unreported
    (runner.py run_stream); the kernel ends the episode on it and reports that decision after the
    snapshot (turns.py TurnPolicy.on_end). A failed token step while pacing reports the decision
    with its own status and the boundary it reached, as today's loop would have written them
    (runner.py _apply)."""
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
    # In model-input frames, n / B, and 0 when paused (turns.py TurnPolicy._accept).
    delay = 0.0 if script.tokens_per_frame is None else tokens / script.tokens_per_frame
    if d["resolved"] is None:
        # The step to the action time failed: the action is not landed (turns.py
        # TurnPolicy._approached).
        tau = t_k + delay * frame_ticks  # runner.py _apply
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
            "token_delay_frames": delay,
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
        "token_delay_frames": delay,
        "action_output_tokens": tokens,
        "post_horizon_output_tokens": 0,
        "frame_age_s": round((tau - t_k) / script.env_fps, 6),
    }


def _whole_up(value: float) -> int:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) < 1e-9 else math.ceil(value)


def _krel(script: Script, p: Projection, notes: dict) -> None:
    """A wall hold's lease starts when the kernel sends Execute (turns.py TurnPolicy._pre_checked),
    and a press the environment's guards delayed is held at least min(duration, 1/60 s) more
    (decision 38, N9); today the lease starts when the environment presses the keys
    (sandbox/actions.py GuestEnvironment._execute_wall_hold)."""
    if not _wall(script) or not script.execute_s:
        return
    for k, sent in p.facts["executed_at"].items():
        word = parse(script.turns[k].action)
        if word.kind != "hold" or not word.seconds:
            continue
        pressed = sent + script.execute_s
        for released in p.episode["key_ups"]:
            if _same(released["at"], pressed + word.seconds):
                lease = max(sent + word.seconds, pressed + min(word.seconds, TAP_HOLD_S))
                released["at"] = round(lease, 9)


def _read_under_way(script: Script, at: float) -> float | None:
    """Wall clock, reads taking time: when the kernel's periodic read under way at ``at`` was
    asked, if one is (turns.py TurnPolicy._schedule_end_check); None if none is. At the instant a
    read is asked, what else falls due goes first (Priority.COMPLETION before START)."""
    if not script.check_s:
        return None
    asked = _floor(at * script.env_fps) / script.env_fps
    if asked <= 0 or not earlier(asked, at) or not earlier(at, asked + script.check_s):
        return None
    return asked


def _read_across(script: Script, p: Projection, at: float) -> float | None:
    """Wall clock: when today's state read asked before ``at`` and answered at it or after was
    asked, of those the kernel makes too (WallEnvironment._kernels); None if none was."""
    for asked, _, kernels in p.facts["reads"]:
        if kernels and earlier(asked, at) and not earlier(asked + script.check_s, at):
            return asked
    return None


def _deadline_handled(script: Script, p: Projection) -> float:
    """Wall clock: when the kernel handles the deadline, once the shell's synchronous operation
    under way at it has returned (evaluation/shell.py _Shell._loop runs a step's commands before
    the next step): its periodic read (D47), a read today's loop makes too, a capture, or an
    action's execution."""
    deadline = script.duration_s
    asked = _read_under_way(script, deadline)
    if asked is None:
        asked = _read_across(script, p, deadline)
    if asked is not None:
        return asked + script.check_s
    for capture in p.episode["captures"]:
        returned = capture["timestamp"]
        if earlier(returned - script.capture_s, deadline) and not earlier(returned, deadline):
            return returned
    for d in p.decisions.values():
        for executed in d["executed"]:
            returned = executed["at"] + script.execute_s
            if earlier(executed["at"], deadline) and not earlier(returned, deadline):
                return returned
    return deadline


def _move_landing(script: Script, p: Projection, k: int, landed: float, delta: float) -> None:
    """Decision k's action, today landed at ``landed``, lands ``delta`` s later, and what runs from
    its landing with it: the hold's lease (K-rel), a wait the deadline cuts, and an end found at
    the action or by the read right after it."""
    d, fps = p.decisions[k], script.env_fps
    moved = landed + delta
    for executed in d["executed"]:
        if _same(executed["at"], landed):
            executed["at"] = round(moved, 9)
    # An action resolves, and its decision paces on, once the read after it has returned.
    after = script.execute_s + script.check_s
    for record, name in ((d["resolved"], "environment_tick"), (d["boundary"], "from")):
        if record is not None and record[name] == _floor((landed + after) * fps):
            record[name] = _floor((moved + after) * fps)
    word = parse(script.turns[k].action)
    if word.kind == "hold":
        for released in p.episode["key_ups"]:
            if _same(released["at"], landed + word.seconds):
                released["at"] = round(released["at"] + delta, 9)
    record = d["wait_record"]
    requested = None if record is None else record.get("requested_duration_simulated_seconds")
    if requested is not None and earlier(script.duration_s, landed + requested):
        record["duration_simulated_seconds"] = round(
            record["duration_simulated_seconds"] - delta, 9
        )
    termination = p.episode["termination"]
    if _same(termination["end_s"], landed + script.execute_s):
        termination["end_s"] = round(termination["end_s"] + delta, 9)
        if termination["snapshot_s"] is not None:
            termination["snapshot_s"] = round(termination["snapshot_s"] + delta, 9)


def _kb_landing(script: Script, p: Projection, notes: dict) -> None:
    """Wall clock, reads taking time: the kernel reads the task every 1/env_fps s during a
    generation too (K-b); a result back while such a read is under way is taken once it has
    returned, so its action, and the frame age and action time it is recorded with, come up to a
    read's time later (N29; evaluation/shell.py _Shell._env). Today's loop does not read during a
    generation."""
    if not _wall(script) or not script.check_s:
        return
    fps = script.env_fps
    for k, d in sorted(p.decisions.items()):
        back = p.facts["finished_at"].get(k)
        asked = None if back is None else _read_under_way(script, back)
        if asked is None or d["approach"] is None:
            continue
        taken = asked + script.check_s
        delay = taken - back
        approach = d["approach"]
        approach["requested"] = round(approach["requested"] + delay * fps, 9)
        if approach["to"] == _floor(back * fps):
            approach["to"] = _floor(taken * fps)
        for record in (d["resolved"], d["outcome"]):
            record["action_clock_position"] = round(
                record["action_clock_position"] + delay * fps, 9
            )
        d["outcome"]["frame_age_s"] = round(d["outcome"]["frame_age_s"] + delay, 6)
        # Both read before landing, the kernel once the read under way has returned (N26).
        landed = p.facts["executed_at"].get(k)
        if landed is not None:
            _move_landing(script, p, k, landed, delay)


def _kk(script: Script, p: Projection, notes: dict) -> None:
    """On the wall clock the frame count runs out at the deadline. After the read before landing
    today's loop asks the deadline, then the frame count (legacy_loop.py _land_action), so a read
    asked before the deadline that answers at it or after leaves the decision
    not_applied_tick_limit; the kernel asks the deadline only (turns.py TurnPolicy._pre_checked,
    _past_deadline), and the deadline's timer settles the decision as not_applied_time_limit
    (TurnPolicy.on_end, NOT_APPLIED)."""
    if not _wall(script):
        return
    asked = _read_across(script, p, script.duration_s)
    if asked is None:
        return
    for k, d in p.decisions.items():
        resolved = d["resolved"]
        if resolved is None or resolved["status"] != "not_applied_tick_limit":
            continue
        if _same(p.facts["finished_at"][k], asked):
            for record in (resolved, d["commit"], d["outcome"]):
                record["status"] = "not_applied_time_limit"


def _u2(script: Script, p: Projection, notes: dict) -> None:
    """The kernel writes RunInvalidated for every failure (core.py Kernel.invalidate), also one the
    environment latched itself; today only the runner's own _invalidate writes it
    (runner.py _invalidate). On the list by N5."""
    if p.episode["invalid"] is not None and p.episode["invalidation_records"] == 0:
        p.episode["invalidation_records"] = 1


def _x2(script: Script, p: Projection, notes: dict) -> None:
    """The kernel freezes the snapshot before it reports the decision in hand (core.py Kernel.end,
    turns.py TurnPolicy.on_end); today the decision is reported first and the snapshot read after
    the loop (runner.py _apply, run_stream)."""
    k = notes["in_hand"] if "in_hand" in notes else _old_in_hand(script, p, notes)
    if k is not None and k in p.decisions and p.decisions[k]["commit"] is not None:
        p.episode["snapshot_after"] -= 1


def _drop_replies(p: Projection, k: int) -> None:
    """Decision k's reply records, and the usage its reply added."""
    trace = p.beyond["trace"]
    trace["OutputChunkReceived"].pop(k, None)
    for completed in trace["InferenceCompleted"].pop(k, []):
        for name, value in (completed["frame_usage"] or {}).items():
            p.beyond["usage"][name] -= value


def _x6(script: Script, p: Projection, notes: dict) -> None:
    """Wall clock: a move the per-move cap gave up on changes no earlier record; its feedback,
    which names no action, opens every next frame until a result is accepted (H3; decision 10;
    harnesses/default/harness.py DefaultHarness.commit). Today the report goes to the newest
    record, and with none it is lost (DefaultHarness.commit_action). Only moves with no accepted
    result before them are rewritten here, as in the named script."""
    if not _wall(script):
        return
    note = f"EXECUTION RESULT: status=skipped_timeout. {UNANSWERED_STEP_NOTES['skipped_timeout']}\n"
    skipped = {
        k
        for k, reason in p.facts["timeout_reasons"].items()
        if reason == "per_move_cap" and p.decisions[k]["commit"]["status"] == "skipped_timeout"
    }
    if not skipped or min(skipped) != 0:
        return
    for request, (k, _) in zip(p.beyond["requests"], p.facts["requested"], strict=True):
        unanswered = sum(1 for j in range(k) if j in skipped)
        if unanswered != k or not unanswered:
            continue
        frame = next(m for m in reversed(request["body"]["messages"]) if m["role"] == "user")
        frame["content"][0]["text"] = note * unanswered + frame["content"][0]["text"]


def _x7(script: Script, p: Projection, notes: dict) -> None:
    """A done or fail the kernel ran as a wait is reported as the wait it ran, with a note
    (H4; harnesses/default/harness.py DefaultHarness._report); today the report names the done."""
    terminal = re.compile(r"EXECUTION RESULT: actual=(done|fail); requested=\1; status=wait\.")
    note = UNSUPPORTED_TERMINAL_ACTION_FEEDBACK

    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return terminal.sub(
                lambda m: f"EXECUTION RESULT: actual=wait; requested={m[1]}; status=wait. {note}",
                value,
            )
        return value

    p.beyond["requests"] = rewrite(p.beyond["requests"])


def _old_in_hand(script: Script, p: Projection, notes: dict) -> int | None:
    """The decision today's loop committed after the first end signal: an end or a failure found
    before its commit, or the wall deadline passing while it paced or waited, or while an
    operation before its resolution was under way. One whose read after the action was asked
    before the deadline the kernel reports first: the answer is filed at its asking
    (core.py Kernel.step), and no pacing follows it past the deadline."""
    committed = p.facts["committed_at"]
    if not committed:
        return None
    last = max(committed)
    if p.facts["ended_before_commit"].get(last):
        return last
    if _ended_by_the_deadline(script, p, notes) and not earlier(committed[last], script.duration_s):
        asked = _read_across(script, p, script.duration_s)
        executed = p.facts["executed_at"].get(last)
        if asked is None or executed is None or earlier(asked, executed):
            return last
    return None


RULES = (
    Rule("X4", "ticks and frame positions agree within 1e-9", _x4),
    Rule("KF2", "a horizon within 1e-9 above a whole tick is that tick", _kf2),
    Rule("KF5", "the state is read at most once a frame, and once after every action", _kf5),
    Rule("KF4", "a tap's read that saw the end ends the episode", _kf4),
    Rule("X1", "a wait is not executed by the environment; its record has no holds", _x1),
    Rule("X12", "the third failed state read in a row invalidates the run", _x12),
    Rule("K-b", "the wall clock checks the end every 1/env_fps s", _kb),
    Rule(
        "KF10",
        "a wall action lands at the reading after the shell accepted its result",
        seen="not observable: the replay accepts in no time; the landing timer's records are"
        " KernelRecords (X5)",
    ),
    Rule(
        "KF5d",
        "a wall action in a frame already read lands before the end is read",
        seen="no longer: the kernel reads before landing on a new reading (N26 (b))",
    ),
    Rule("K-c", "a wall wait is a timer the end interrupts", _kc),
    Rule("K-i", "think budget and the task's end or the deadline: the first names the reason", _ki),
    Rule("K-d", "nothing advances after the first end signal", _kd),
    Rule("K-a", "the wall deadline is a timer without grace", _ka),
    Rule("D47", "a result back during a read the deadline falls in is filed after it", _d47),
    Rule("U10", "a generation the task's end gave up on is recorded as given up", _u10),
    Rule("U11", "a generation an invalidation gave up on is recorded as given up", _u11),
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
    Rule("K-b", "a result back during a periodic read is taken once it returns", _kb_landing),
    Rule(
        "U14",
        "a failed wall read invalidates the run when it returns, not when asked",
        seen="no longer: N44 dates the invalidation by the read's asking, as today's loop does",
    ),
    Rule("K-k", "the wall clock does not compare the frame count separately", _kk),
    Rule(
        "K-l",
        "the per-move cap runs from StartGeneration",
        seen="not observable: no time passes between the capture and _run_step here",
    ),
    Rule(
        "K-poll",
        "waiting for the next wall frame is a timer, not a 30 ms poll",
        seen="not observable: FakeEnvironment lands on whole ticks (fake_env.py wait_until_tick)",
    ),
    Rule("U2", "an environment-latched failure is written as RunInvalidated", _u2),
    Rule(
        "N22",
        "a failed wait step's error is the traceback",
        seen="normalized to its message in projection.wait_record, for the simulation too",
    ),
    Rule(
        "U7",
        "a result of a step over the per-move cap that is unusable is reported skipped_timeout",
        seen="not scripted: no replayed trajectory row fails to be written (p1b 第四轮 L3)",
    ),
    Rule(
        "U8",
        "a frozen frame after another invalidation is written as captured",
        seen="not scripted: no replayed wall release fails while a frame is captured",
    ),
    Rule(
        "U9",
        "a result settled past the per-move cap, or filed late, is skipped_timeout",
        seen="not scripted: the replay settles a result in no time, and no named script has a"
        " read under way when the cap falls (D47 covers the deadline)",
    ),
    Rule(
        "KF7",
        "a result back before the deadline and settled past it is discarded",
        seen="not scripted: the replay settles a result in no time",
    ),
    Rule(
        "U13",
        "a result settled and then discarded writes its trajectory row as a decision's",
        seen="not compared: the trajectory rows (p1b 第七轮 U13)",
    ),
    Rule(
        "U12",
        "the records of an exception the shell caught",
        seen="not scripted: no replayed kernel, policy or state reading raises",
    ),
    Rule("X2", "the snapshot is frozen before the decision in hand is reported", _x2),
    Rule(
        "X3",
        "a generation given up on leaves no reply records",
        seen="no longer: the shell records it as a call given up on (decision 48, N10; K-a, U10,"
        " U11)",
    ),
    Rule(
        "X5",
        "KernelRecord entries in the episode trace",
        seen="left out of the projection, as the plan's 投影 says",
    ),
    Rule("X6", "a timed-out step's feedback opens the next frame, no record changes", _x6),
    Rule("X7", "DONE turned into a wait is reported as actual=wait", _x7),
    Rule(
        "X8",
        "generation numbers in the agent trace; Decision client rebuilt",
        seen="not compared: the agent's event log, and no Decision API is replayed here",
    ),
    Rule("X9", "agent events numbered by the kernel", seen="P1d"),
    Rule(
        "X17",
        "token ticks are equal floats where today's are integers",
        seen="not compared: the projection compares numbers by value",
    ),
    Rule("X10", "SessionReleaseWait agent records", seen="P3, live drivers only"),
    Rule(
        "X11",
        "token key events and releases bounded by the environment budget",
        seen="both loops run the same environment; replayed key events take no time",
    ),
    Rule(
        "X13",
        "a wall state read is bounded by 10 s",
        seen="not observable: the bound is GuestEnvironment.check_end's, and the replayed wall"
        " environment is a FakeEnvironment that bounds no read; tests/sandbox/"
        "test_shell_environment.py covers it",
    ),
    Rule(
        "X14",
        "an action's whole budget is 20 s",
        seen="both loops run the same environment; replayed actions take the script's time",
    ),
    Rule(
        "X15",
        "a wait for the wall's key commands ends with the action's budget; key checks and hold"
        " snapshots do not wait for them",
        seen="both loops run the same environment; no replayed lock is contended",
    ),
    Rule(
        "X16",
        "a failed key release at cleanup leaves the run as it was",
        seen="not scripted: no replayed release fails; both loops only log one (legacy_loop.py"
        " _finalize, shell.py _release_inputs), which tests/shell and tests/shell_streaming cover"
        " for the shells",
    ),
)
ORDER = [rule.id for rule in RULES]
