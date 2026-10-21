"""Named differential scripts (wave2-plan.md, P4 具名场景).

1. The frame-rate cases of tests/replay/test_runner_frame_rates.py (TOKEN_CASES, WALL_CASES),
   with their replies, token counts and latencies as they are.
2. The runner-loop cases of tests/replay/test_runner_loop.py (CASES and the long scenario, its
   actions spelled in the script vocabulary).
3. What the replay tests lacked: a wall result exactly at the deadline, a wall wait across the
   deadline, a retry across the deadline, a hold expiring between two frames on both clocks, a
   frozen stream.
4. A script for each expected difference (rules.py).
5. phase2.md:283: a wall game that ends during a model call, found by the periodic check; an
   early end whose teardown runs past the deadline.

Each script's ``expect`` lists the differences it is meant to show.
"""

from __future__ import annotations

from tests.replay.scenario import OUTSIDE_PAGE, UNSUPPORTED_KEY

from .script import Attempt, Script, Turn

SHORT_S = 3.75  # test_runner_loop's short episode: decisions at 0, 1, 2 and 3 s at 1 fps
LATE_REPLY_S = 0.875  # a reply to the decision at 3.0 s that lands after the deadline
HANG = (Attempt(hang=True),)


def _timed(words: list[str], latencies: list[float], **fields) -> tuple[Turn, ...]:
    return tuple(
        Turn(word, attempts=(Attempt(latency),), **fields)
        for word, latency in zip(words, latencies, strict=True)
    )


def _tokens(pairs: list[tuple[str, int]]) -> tuple[Turn, ...]:
    return tuple(Turn(word, tokens) for word, tokens in pairs)


# ---- 1. the frame-rate cases ------------------------------------------------------------------

FRAME_RATES = [
    Script(
        "token",
        5.625,
        _tokens(
            [("key:tab", 80), ("click", 240), ("click", 160), ("wait:0.5", 32), ("click", 130)]
        ),
        env_fps=60.0,
        tokens_per_frame=160,
        expect=frozenset({"KF5", "X1"}),
        name="frame_rates-token-60_1",
    ),
    Script(
        "token",
        1.875,
        _tokens(
            [("click", 35), ("key:tab", 140), ("wait:0.1", 70), ("click", 139), ("click", 280)]
        ),
        env_fps=30.0,
        input_fps=7.0,
        tokens_per_frame=70,
        expect=frozenset({"KF5", "X1"}),
        name="frame_rates-token-30_7",
    ),
    Script(
        "token",
        1.875,
        _tokens([("key:tab", 100), ("wait:0.25", 50), ("click", 200)]),
        env_fps=2.0,
        input_fps=2.0,
        tokens_per_frame=100,
        expect=frozenset({"KF5", "X1"}),
        name="frame_rates-token-2_2",
    ),
    Script(
        "pause",
        3.75,
        _tokens([("click", 80), ("wait:1.5", 40), ("click", 500)]),
        env_fps=60.0,
        expect=frozenset({"KF5", "X1"}),
        name="frame_rates-token-pause-60_1",
    ),
    Script(
        "wall",
        3.75,
        _timed(["click", "type:Ada", "click"], [0.5, 1.25, 0.875]),
        env_fps=60.0,
        expect=frozenset({"K-a", "K-g"}),
        name="frame_rates-wall-60_1",
    ),
    Script(
        "wall",
        0.9375,
        _timed(["click", "type:Ada", "click", "click"], [0.1, 0.2, 0.05, 0.5]),
        env_fps=30.0,
        input_fps=7.0,
        expect=frozenset({"K-a", "K-g", "X4"}),
        name="frame_rates-wall-30_7",
    ),
    Script(
        "wall",
        1.875,
        _timed(["click", "type:Ada", "click"], [0.25, 0.75, 0.5]),
        env_fps=2.0,
        input_fps=2.0,
        expect=frozenset({"K-a", "K-g"}),
        name="frame_rates-wall-2_2",
    ),
]

# ---- 2. the runner-loop cases -----------------------------------------------------------------

FORM = ["click", "type:Ada", "click", "click"]

RUNNER_LOOP = [
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5, 0.5, 0.5, LATE_REPLY_S]),
        expect=frozenset({"K-a", "K-g"}),
        name="runner-late_reply",
    ),
    Script(
        "wall",
        SHORT_S,
        _timed(["click", "done", "click", "click"], [0.5, 0.5, 0.5, LATE_REPLY_S]),
        expect=frozenset({"K-a", "K-g", "X1", "X7"}),
        name="runner-done_reply",
    ),
    Script(
        "wall",
        SHORT_S,
        _timed(FORM[:3], [0.5, 0.5, 0.5]) + (Turn("click"),),
        ends_after=3,
        end_reason="task_completed",
        expect=frozenset({"X2"}),
        name="runner-environment_done",
    ),
    Script(
        "wall",
        SHORT_S,
        _timed(FORM[:3], [0.5, 0.5, 0.5]) + (Turn("click"),),
        ends_at_s=2.25,
        expect=frozenset({"KF5d", "X2"}),
        name="runner-environment_done_during_reply",
    ),
    Script(
        "wall",
        SHORT_S,
        _timed(FORM[:3], [0.5, 0.5, 0.5]) + (Turn("click", attempts=HANG),),
        expect=frozenset({"K-a", "K-g"}),
        name="runner-bounded_call_timeout",
    ),
    Script(
        "wall",
        SHORT_S,
        (Turn("click"), Turn("type:Ada", fault="guest connection reset"), Turn("click")),
        expect=frozenset({"U2", "X2"}),
        name="runner-environment_failure",
    ),
]

# test_runner_loop's LONG_SCENARIO in the vocabulary: pointer actions are clicks (no drag, scroll
# or move duration reaches the kernel), and a reply with no single valid action is garbage.
LONG_WORDS = [
    # 0-9
    "click", "type:Ada Lovelace", "key:tab", "click", "click", "wait:1", "garbage", "key:hyper",
    "key:ctrl+a", "click",
    # 10-19
    "click", "click", "click", "hold:shift:0.5", "garbage", "click", "click", "garbage",
    "garbage", "wait",
    # 20-29
    "click", "click", "type:Ada", "click", "key:enter", "click", "click", "key:backspace:3",
    "click", "click",
    # 30-39
    "click", "click", "click", "wait:0.5", "key:escape", "key:tab", "type:Lovelace", "click",
    "garbage", "hold:shift+tab:0.2",
    # 40-49
    "click", "click", "click", "garbage", "key:ctrl+z", "click", "click", "wait", "click",
    "click",
    # 50-59
    "type:Ada Lovelace", "click", "click", "key:hyper", "click", "click", "wait:2", "garbage",
    "click", "click",
]  # fmt: skip
LONG_REJECTED = {7: UNSUPPORTED_KEY, 21: OUTSIDE_PAGE, 53: UNSUPPORTED_KEY}
LONG_LATENCIES = [
    {3: 0.25, 8: 0.75, 12: 2.5, 21: 1.0, 26: 0.25, 40: 1.75, 47: 0.75, 59: 0.875}.get(k, 0.5)
    for k in range(len(LONG_WORDS))
]
LONG = Script(
    "wall",
    63.75,
    tuple(
        Turn(word, attempts=(Attempt(latency),), rejected=LONG_REJECTED.get(k))
        for k, (word, latency) in enumerate(zip(LONG_WORDS, LONG_LATENCIES, strict=True))
    ),
    expect=frozenset({"K-a", "K-g", "X1"}),
    name="runner-long",
)

# ---- 3. what the replay tests lacked ----------------------------------------------------------

MISSING = [
    # Decision 3 starts at 3.0 s and its reply comes exactly at the 3.75 s deadline.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5, 0.5, 0.5, 0.75]),
        expect=frozenset({"K-a"}),
        name="wall-result-at-the-deadline",
    ),
    # Decision 2 starts a 5 s wait at 2.5 s; the deadline cuts it at 3.75 s.
    Script(
        "wall",
        SHORT_S,
        _timed(["click", "type:Ada", "wait:5"], [0.5, 0.5, 0.5]),
        expect=frozenset({"X1", "X2"}),
        name="wall-wait-across-the-deadline",
    ),
    # Decision 3's request fails twice with HTTP 503 at once, as in test_runner_clock: the first
    # backoff (1 s +-25%, entity/model.py) already runs past the 3.75 s deadline, and today's
    # bounded wait gives up at 5.75 s. The simulation needs only a reply past the deadline.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM[:3], [0.5] * 3)
        + (Turn("click", attempts=(Attempt(0.0, 503), Attempt(0.0, 503), Attempt(0.5))),),
        expect=frozenset({"K-a", "K-g"}),
        name="wall-retry-across-the-deadline",
    ),
    # A 0.5 s hold pressed at 0.25 s is released at 0.75 s, between the frames at 0 and 1 s.
    Script(
        "wall",
        2.5,
        _timed(["hold:shift:0.5", "click", "click"], [0.25, 0.25, 0.25]),
        expect=frozenset({"X2"}),
        name="wall-hold-between-frames",
    ),
    # A 0.5 s hold at tick 0 on a 1 fps token clock ends at tick 0.5, between frames 0 and 1.
    Script(
        "token",
        3.0,
        _tokens([("hold:a:0.5", 0), ("click", 50), ("click", 50)]),
        tokens_per_frame=100,
        expect=frozenset({"KF5"}),
        name="token-hold-between-frames",
    ),
    # From capture 1 on every frame repeats; the third identical one invalidates the run.
    Script(
        "wall",
        6.0,
        _timed(["click"] * 5, [0.5] * 5),
        frozen_from=1,
        expect=frozenset({"K-h"}),
        name="wall-frozen-stream",
    ),
    Script(
        "token",
        6.0,
        _tokens([("click", 50)] * 5),
        tokens_per_frame=100,
        frozen_from=1,
        expect=frozenset({"K-h", "KF5"}),
        name="token-frozen-stream",
    ),
]

# ---- 4. a script for each expected difference --------------------------------------------------

TWO_CLICKS = _tokens([("click", 50)] * 4)

PER_RULE = [
    # K-b: at 4 fps the check at 2.25 s finds the end while decision 2 waits for its reply.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        env_fps=4.0,
        ends_at_s=2.25,
        expect=frozenset({"K-b", "K-g", "U10"}),
        name="wall-ends-during-a-reply",
    ),
    # K-b: the check at 0.75 s finds the end while decision 0 paces to the frame at 1 s.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.25] * 4),
        env_fps=4.0,
        ends_at_s=0.6,
        expect=frozenset({"K-b", "K-g", "X2"}),
        name="wall-ends-while-pacing",
    ),
    # K-c: the check at 2 s finds the end in the middle of a 2.5 s wait started at 0.25 s.
    Script(
        "wall",
        SHORT_S,
        _timed(["wait:2.5", "click", "click"], [0.25] * 3),
        ends_at_s=1.5,
        expect=frozenset({"K-b", "K-c", "K-g", "X1", "X2"}),
        name="wall-wait-cut-by-the-end",
    ),
    # K-rel: the hold takes effect 0.125 s after the Execute, once the guards and focus ran.
    Script(
        "wall",
        2.5,
        _timed(["hold:shift:0.5", "click", "click"], [0.25] * 3),
        execute_s=0.125,
        expect=frozenset({"K-rel", "X2"}),
        name="wall-hold-pressed-after-the-guards",
    ),
    # K-i: decision 1 runs out of the think budget and skips its action; the check at 1.75 s
    # finds the task's end while it paces.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        env_fps=4.0,
        think_budget_s=0.75,
        ends_at_s=1.75,
        expect=frozenset({"K-g", "K-i", "X2"}),
        name="wall-think-budget-and-the-end",
    ),
    # The per-move cap gives up on a 0.5 s reply after 0.4 s. K-l does not show: no time passes
    # between the capture and the generation's start (rules.py).
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        per_move_cap_s=0.4,
        expect=frozenset({"X2", "X4", "X6"}),
        name="wall-per-move-cap",
    ),
    # K-f: decision 1's reply reports no usage on the finite token clock.
    Script(
        "token",
        3.0,
        (Turn("click", 50), Turn("click", None), Turn("click", 50)),
        tokens_per_frame=100,
        expect=frozenset({"K-f", "KF5"}),
        name="token-missing-usage",
    ),
    # K-j: a key held until tick 1.495 is still down when decision 1 taps at tick 1.49; the
    # tap's 1/60 s outlasts the lease.
    Script(
        "token",
        3.0,
        _tokens([("hold:a:1.495", 0), ("key:tab", 49), ("click", 50), ("click", 50)]),
        tokens_per_frame=100,
        expect=frozenset({"K-j", "KF5"}),
        name="token-lease-ends-inside-a-tap",
    ),
    # K-e: the first clock step, decision 0's move to its action time 0.5, fails.
    Script(
        "token",
        3.0,
        TWO_CLICKS,
        tokens_per_frame=100,
        advance_errors=(1,),
        expect=frozenset({"K-e", "U2"}),
        name="token-advance-fails-while-approaching",
    ),
    # K-e: the second clock step, decision 0's pacing from 0.5 to 1, fails.
    Script(
        "token",
        3.0,
        TWO_CLICKS,
        tokens_per_frame=100,
        advance_errors=(2,),
        expect=frozenset({"K-e", "KF5", "U2"}),
        name="token-advance-fails-while-pacing",
    ),
    # K-d: decision 0's 384 tokens are 1.5 s, past the 1 s per-move cap, so it places no action;
    # the task ends at tick 1, found on the way to the action time. Today the decision still
    # paces to the frame at tick 2.
    Script(
        "token",
        4.0,
        _tokens([("click", 384)] + [("click", 50)] * 3),
        tokens_per_frame=256,
        per_move_cap_s=1.0,
        ends_at_s=1.0,
        expect=frozenset({"K-d", "KF5", "X2"}),
        name="token-skipped-decision-after-the-end",
    ),
    # K-i on the token clock: the same, with the think budget run out instead of the cap.
    Script(
        "token",
        4.0,
        _tokens([("click", 384)] + [("click", 50)] * 3),
        tokens_per_frame=256,
        think_budget_s=0.5,
        ends_at_s=1.0,
        expect=frozenset({"K-d", "K-i", "KF5", "X2"}),
        name="token-think-budget-and-the-end",
    ),
    # The task ends at 0.75 s, found by the check after the pacing step to tick 1.
    Script(
        "token",
        3.0,
        TWO_CLICKS,
        tokens_per_frame=100,
        ends_at_s=0.75,
        expect=frozenset({"KF5", "X2"}),
        name="token-ends-while-pacing",
    ),
]

# X12: the kernel invalidates on the third failed state read in a row, today's loop on the first.
# Token clock, 1/1, B = 100, 50-token actions. The kernel reads at most once a frame and after
# every action (decision 31, KF5): decision 0 reads at the loop top (check 1), after the step to
# its action time 0.5 (2), after the action (3) and after pacing to tick 1 (4); decision 1 reads
# after its step to 1.5 (5), and so on. A tap at 0.5 reads once inside, at 0.5 + 1/60 (tap read
# 1), between 2 and 3. Today's loop also reads before landing and at the loop top on a tick
# already read; those reads go unnumbered (replayed.TokenWorld._count_read).
TAP_FIRST = _tokens([("key:tab", 50)] + [("click", 50)] * 3)

X12 = [
    # Checks 2 and 3 fail and 4 reads: the kernel goes on, today's loop stops at check 2.
    Script(
        "token",
        3.0,
        TWO_CLICKS,
        tokens_per_frame=100,
        check_errors=(2, 3),
        expect=frozenset({"KF5", "X12"}),
        name="token-two-failed-reads-in-a-row-are-tolerated",
    ),
    # Checks 1 to 3 fail; the third is the one after decision 0's click, which stands applied with
    # the episode over after it. The kernel never reads before landing on the token clock (KF5b),
    # so K-n is not seen here.
    Script(
        "token",
        3.0,
        TWO_CLICKS,
        tokens_per_frame=100,
        check_errors=(1, 2, 3),
        expect=frozenset({"KF5", "U2", "X12", "X2"}),
        name="token-third-failed-read-after-the-action",
    ),
    # Checks 1 and 2 fail and so does the read inside the tap: the tap is reported applied, with
    # the episode over after it.
    Script(
        "token",
        3.0,
        TAP_FIRST,
        tokens_per_frame=100,
        check_errors=(1, 2),
        tap_check_errors=(1,),
        expect=frozenset({"KF5", "U2", "X12", "X2"}),
        name="token-third-failed-read-inside-a-tap",
    ),
    # Checks 1 and 2 fail, the read inside the tap succeeds and resets the count, and checks 3
    # and 4 fail: two in a row again, so the run goes on.
    Script(
        "token",
        3.0,
        TAP_FIRST,
        tokens_per_frame=100,
        check_errors=(1, 2, 3, 4),
        expect=frozenset({"KF5", "X12"}),
        name="token-a-read-inside-a-tap-resets-the-count",
    ),
    # The read inside the tap fails, then checks 3 and 4: the third is the check after pacing.
    Script(
        "token",
        3.0,
        TAP_FIRST,
        tokens_per_frame=100,
        check_errors=(3, 4),
        tap_check_errors=(1,),
        expect=frozenset({"KF5", "U2", "X12", "X2"}),
        name="token-a-failed-read-inside-a-tap-counts",
    ),
    # KF4: the task ends at 0.51 s, between check 2 and the read inside the tap, which sees the
    # end: the kernel ends the episode there, by the task's reason, and does not read after the
    # tap; today's loop does.
    Script(
        "token",
        3.0,
        TAP_FIRST,
        tokens_per_frame=100,
        ends_at_s=0.51,
        expect=frozenset({"KF4", "KF5", "X2"}),
        name="token-the-end-seen-inside-a-tap",
    ),
    # As above, with check 3 to fail: it would be the read after the tap, which the kernel never
    # makes, so nothing fails. X12's former note c, an end found late after it, is gone.
    Script(
        "token",
        3.0,
        TAP_FIRST,
        tokens_per_frame=100,
        ends_at_s=0.51,
        check_errors=(3,),
        expect=frozenset({"KF4", "KF5", "X2"}),
        name="token-no-read-after-the-end-seen-inside-a-tap",
    ),
    # At 240 fps the tap at tick 0.5 steps to 1.5, 2.5, 3.5 and 4.5, reading after each step.
    # The third failure is its first read: the kernel's tap goes on to 4.5, today's stops at 1.5.
    Script(
        "token",
        3.0,
        TAP_FIRST,
        env_fps=240.0,
        input_fps=240.0,
        tokens_per_frame=100,
        check_errors=(1, 2),
        tap_check_errors=(1,),
        expect=frozenset({"KF5", "U2", "X12", "X2"}),
        name="token-a-tap-goes-on-after-the-third-failed-read",
    ),
    # As above, with the task over at tick 2: the tap's read at 2.5 sees it and stops the tap.
    Script(
        "token",
        3.0,
        TAP_FIRST,
        env_fps=240.0,
        input_fps=240.0,
        tokens_per_frame=100,
        check_errors=(1, 2),
        tap_check_errors=(1,),
        ends_at_s=2.0 / 240.0,
        expect=frozenset({"KF5", "U2", "X12", "X2"}),
        name="token-a-tap-goes-on-after-the-third-failed-read-to-the-end",
    ),
    # Wall clock, decisions at 0, 1, 2 and 3 s. The kernel reads each frame by its timer and after
    # every action (decision 31). At 3 fps its only read within [1.4, 1.6) is the one after
    # decision 1's click at 1.5 s; the periodic read at 5/3 s succeeds.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        env_fps=3.0,
        failing_reads_s=((1.4, 1.6),),
        expect=frozenset({"X12", "X2"}),
        name="wall-two-failed-reads-are-tolerated",
    ),
    # At 8 fps the periodic reads at 1.25 and 1.375 s fail, then the one after decision 1's
    # click at 1.45 s, the third. Today's loop makes no periodic reads, so its first failed read
    # is that one too, and X12 changes nothing.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5, 0.45, 0.5, 0.5]),
        env_fps=8.0,
        failing_reads_s=((1.2, 1.6),),
        expect=frozenset({"U2", "X2"}),
        name="wall-third-failed-read-is-todays-first",
    ),
    # The periodic reads at 1.125, 1.25 and 1.375 s fail while decision 1 waits for its reply:
    # the kernel gives up on the generation; today's loop never reads then.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        env_fps=8.0,
        failing_reads_s=((1.1, 1.45),),
        expect=frozenset({"K-g", "U11", "X12"}),
        name="wall-third-failed-read-during-a-reply",
    ),
    # Two failed periodic reads, a successful one at 1.375 s, then the one after decision 1's
    # click at 1.45 s fails: never three in a row.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5, 0.45, 0.5, 0.5]),
        env_fps=8.0,
        failing_reads_s=((1.1, 1.3), (1.4, 1.48)),
        expect=frozenset({"X12", "X2"}),
        name="wall-a-successful-read-resets-the-count",
    ),
]

# A fault the environment latches itself. Both scripts assert today's loop's outcome.
LATCHED = [
    # Former U4: decision 0 waits 2 s from its action time 0.5; its steps go to 1.5 and 2.5, and
    # the second, the third clock step of the episode, fails, so the wait stops at 1.5. The shell
    # answers the step with its error only (evaluation/shell.py _Shell._failure), and the kernel
    # settles the wait on it as a failed action, invalid_run at tick 1.5, as today's environment
    # does (turns.py TurnPolicy._step_failed).
    Script(
        "token",
        4.0,
        _tokens([("wait:2", 50)] + [("click", 50)] * 3),
        tokens_per_frame=100,
        advance_errors=(3,),
        expect=frozenset({"KF5", "U2", "X1", "X2"}),
        name="token-a-wait-whose-own-step-fails",
    ),
    # Former U5: decision 1's click at 1.5 s fails in the environment, which latches the run
    # invalid while execute() returns; reading the snapshot takes 1.5 s. The shell answers the
    # action with its error before the latched Invalidated (N6; evaluation/shell.py
    # _Shell._execute), so the kernel settles the click where it ran, at tick 1, as today's loop.
    Script(
        "wall",
        SHORT_S,
        (
            Turn("click", attempts=(Attempt(0.5),)),
            Turn("click", attempts=(Attempt(0.5),), fault="guest connection reset"),
        ),
        snapshot_s=1.5,
        expect=frozenset({"U2", "X2"}),
        name="wall-a-failed-action-settles-after-the-snapshot-read",
    ),
]

# Named scripts that assert today's outcome where the shell is still to be fixed, with why:
# they fail until then (test_named.py, xfail strict), and pass once it is.
PENDING: dict[str, str] = {}

# KF5d: the task ends at a decision's landing, after its frame was read. The kernel lands the
# action without a read before it; today's loop reads, finds the end and lands nothing.
LANDED = [
    # A 0.5 s wait lands at 0.5 s and runs out at 1 s, where its timer goes before the periodic
    # read, and the read after it finds the end.
    Script(
        "wall",
        1.25,
        _timed(["wait:0.5"], [0.5]),
        ends_at_s=0.5,
        expect=frozenset({"K-g", "KF5d", "X2"}),
        name="wall-a-wait-lands-after-the-end",
    ),
    # A refused click at 0.5 s is read after by nobody; the periodic read at 1 s finds the end
    # while the decision paces.
    Script(
        "wall",
        2.0,
        (Turn("click", attempts=(Attempt(0.5),), rejected="the guest refused the action"),),
        ends_at_s=0.25,
        expect=frozenset({"K-g", "KF5d", "X2"}),
        name="wall-a-refused-action-lands-after-the-end",
    ),
]

# KF2: at 5 fps the horizon main() computes from 0.17 minutes is 51.00000000000001 ticks, which
# the kernel takes as 51. 1000-token replies at B = 100 move ten frames each; decision 5's lands
# on tick 51.
HORIZON = [
    Script(
        "token",
        0.17 * 60,
        tuple(Turn("click", output_tokens=1000) for _ in range(7)),
        env_fps=5.0,
        input_fps=5.0,
        tokens_per_frame=100,
        expect=frozenset({"KF2", "KF5", "X4"}),
        name="token-horizon-a-few-ulps-past-a-tick",
    ),
]

# ---- 5. phase2.md:283 --------------------------------------------------------------------------

EARLY_END = [
    # The third action ends the task at 2.5 s; reading its state takes 1.5 s, past the deadline.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        ends_after=3,
        end_reason="task_success",
        snapshot_s=1.5,
        expect=frozenset({"X2"}),
        name="wall-early-end-snapshot-past-the-deadline",
    ),
]

NAMED = (
    FRAME_RATES
    + RUNNER_LOOP
    + [LONG]
    + MISSING
    + PER_RULE
    + X12
    + LATCHED
    + LANDED
    + HORIZON
    + EARLY_END
)
