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
        expect=frozenset({"K-a", "K-g", "U17"}),
        name="frame_rates-wall-60_1",
    ),
    # The replay's 0.2 s for decision 1 is 0.21 s here: back at 11/30 s, the kernel's landing
    # timer (N25) and its periodic read would fall a few ulps apart, and the fake clock's reading
    # decides whether one step takes both (simulated.ties; R13).
    Script(
        "wall",
        0.9375,
        _timed(["click", "type:Ada", "click", "click"], [0.1, 0.21, 0.05, 0.5]),
        env_fps=30.0,
        input_fps=7.0,
        expect=frozenset({"K-a", "K-g", "U17", "X4"}),
        name="frame_rates-wall-30_7",
    ),
    Script(
        "wall",
        1.875,
        _timed(["click", "type:Ada", "click"], [0.25, 0.75, 0.5]),
        env_fps=2.0,
        input_fps=2.0,
        expect=frozenset({"K-a", "K-g", "U17"}),
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
        expect=frozenset({"K-a", "K-g", "U17"}),
        name="runner-late_reply",
    ),
    Script(
        "wall",
        SHORT_S,
        _timed(["click", "done", "click", "click"], [0.5, 0.5, 0.5, LATE_REPLY_S]),
        expect=frozenset({"K-a", "K-g", "U17", "X1", "X7"}),
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
        expect=frozenset({"X2"}),
        name="runner-environment_done_during_reply",
    ),
    Script(
        "wall",
        SHORT_S,
        _timed(FORM[:3], [0.5, 0.5, 0.5]) + (Turn("click", attempts=HANG),),
        expect=frozenset({"K-a", "K-g", "U17"}),
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
    expect=frozenset({"K-a", "K-g", "U17", "X1"}),
    name="runner-long",
)

# ---- 3. what the replay tests lacked ----------------------------------------------------------

MISSING = [
    # Decision 3 starts at 3.0 s and its reply comes exactly at the 3.75 s deadline.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5, 0.5, 0.5, 0.75]),
        expect=frozenset({"K-a", "U17"}),
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
        expect=frozenset({"K-a", "K-g", "U17"}),
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
        expect=frozenset({"K-b", "K-g", "U10", "U17"}),
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
    # K-rel, decision 38: a 0.1 s hold sent at 0.25 s is pressed at 0.45 s, past the 0.35 s its
    # lease ends; the environment holds it 1/60 s more, to 0.467 s.
    Script(
        "wall",
        2.5,
        _timed(["hold:shift:0.1", "click", "click"], [0.25] * 3),
        execute_s=0.2,
        expect=frozenset({"K-rel", "X2", "X4"}),
        name="wall-a-hold-pressed-past-its-lease",
    ),
    # Key names as the environment knows them: "left" is the arrow key.
    Script(
        "wall",
        2.5,
        _timed(["hold:left:0.5", "click", "click"], [0.25] * 3),
        expect=frozenset({"X2"}),
        name="wall-a-key-under-another-name",
    ),
    Script(
        "token",
        3.0,
        _tokens([("hold:left:0.5", 0), ("click", 50), ("click", 50)]),
        tokens_per_frame=100,
        expect=frozenset({"KF5"}),
        name="token-a-key-under-another-name",
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
    # K-i with the deadline: decision 3's reply at 3.5 s runs out the think budget, and its pacing
    # to the next frame runs into the deadline at 3.75 s, which comes before the budget's end.
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        think_budget_s=1.9,
        expect=frozenset({"K-i", "X2"}),
        name="wall-think-budget-at-the-deadline",
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
    # A move over the per-move cap that also runs out the think budget is skipped_timeout: no
    # action comes before over the budget (turns.py TurnPolicy._skipped, legacy_loop.py
    # _land_action). Every 1.5 s reply is over the 1 s cap and costs the cap; decision 1 brings
    # the spend to 2 s, past the 1.5 s budget.
    Script(
        "wall",
        10.0,
        _timed(["click"] * 4, [1.5] * 4),
        per_move_cap_s=1.0,
        think_budget_s=1.5,
        expect=frozenset({"X6"}),
        name="wall-over-the-cap-and-the-think-budget",
    ),
    # The same on the token clock, B = 256: 768 tokens are 3 s, over the 1 s cap.
    Script(
        "token",
        10.0,
        _tokens([("click", 768), ("click", 768), ("click", 50)]),
        tokens_per_frame=256,
        per_move_cap_s=1.0,
        think_budget_s=1.5,
        expect=frozenset({"KF5"}),
        name="token-over-the-cap-and-the-think-budget",
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
        expect=frozenset({"K-n", "U2", "X2"}),
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
        expect=frozenset({"K-g", "U11", "U17", "X12"}),
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
PENDING: dict[str, str] = {
    "wall-an-action-runs-past-the-deadline-and-the-snapshot-takes-time": (
        "the shell's ActionExecuted has no finished_at, so the kernel files the answer of an action"
        " that ran past the deadline after it has read the snapshot and resolves the action at the"
        " tick then, past the horizon (kernel/core.py Kernel.step, turns.py TurnPolicy._tick);"
        " today's loop resolves it at the tick the action returned at"
    ),
}

# The task ends at a decision's landing, after its frame was read. The kernel reads before
# landing on the new reading (N26 (b)), finds the end and lands nothing, as today's loop does;
# under decision 31's first reading it landed the action (KF5d).
LANDED = [
    # A 0.5 s wait that would land at 0.5 s.
    Script(
        "wall",
        1.25,
        _timed(["wait:0.5"], [0.5]),
        ends_at_s=0.5,
        expect=frozenset({"X2"}),
        name="wall-a-wait-lands-after-the-end",
    ),
    # A click the guest would refuse at 0.5 s.
    Script(
        "wall",
        2.0,
        (Turn("click", attempts=(Attempt(0.5),), rejected="the guest refused the action"),),
        ends_at_s=0.25,
        expect=frozenset({"X2"}),
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

# ---- 6. state reads and captures that take time (N29) --------------------------------------------

TIMED = [
    # At 1 fps each decision reads at k s, captures 0.01 s later and asks 0.02 s after that; its
    # reply is back at k + 0.53 s, and both loops read again before landing it at k + 0.54 s. The
    # last reply misses the deadline (K-a).
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5, 0.5, 0.5, LATE_REPLY_S]),
        check_s=0.01,
        capture_s=0.02,
        expect=frozenset({"K-a", "K-g", "U17"}),
        name="wall-reads-and-captures-take-time",
    ),
    # At 4 fps a read takes 0.05 s. Decision 0's reply is back at 0.52 s, during the periodic read
    # asked at 0.5 s: the kernel takes it at 0.55 s, when that read has returned, and reads again
    # before landing it at 0.6 s (K-b); today's loop takes it at once and lands it at 0.57 s.
    # Decision 1's is back at 1.65 s, between reads.
    Script(
        "wall",
        2.0,
        _timed(["click", "click"], [0.47, 0.6]),
        env_fps=4.0,
        check_s=0.05,
        expect=frozenset({"K-b", "X2", "X4"}),
        name="wall-a-reply-back-during-a-periodic-read",
    ),
    # The deadline at 1.775 s falls during the periodic read asked at 1.75 s, which returns at
    # 1.8 s; decision 1's reply came back at 1.76 s, during it. The shell files the reply after the
    # read, past the deadline (D47), and reads the snapshot then (K-g); today's loop lands it.
    Script(
        "wall",
        1.775,
        _timed(["click", "click"], [0.5, 0.71]),
        env_fps=4.0,
        check_s=0.05,
        expect=frozenset({"D47", "K-g", "U17", "X4"}),
        name="wall-the-deadline-falls-during-a-read",
    ),
    # U9: decision 0 is asked at 0.05 s with a 0.47 s per-move cap, due at 0.52 s; its reply is
    # back at 0.51 s, during the periodic read asked at 0.5 s. The shell files it once the read
    # has returned, at 0.55 s, after the cap: the kernel skips the move, and the next request
    # opens with the timeout's note (X6). Today's loop lands the click.
    Script(
        "wall",
        2.0,
        _timed(["click"] * 4, [0.46, 0.3, 0.3, 0.3]),
        env_fps=4.0,
        check_s=0.05,
        per_move_cap_s=0.47,
        expect=frozenset({"K-b", "U9", "X2", "X4", "X6"}),
        name="wall-the-per-move-cap-falls-during-a-read",
    ),
    # wall-third-failed-read-is-todays-first with reads of 0.01 s: the third failed read, the one
    # after decision 1's click at 1.46 s, invalidates the run dated when it was asked, and the
    # snapshot is read when it has returned, as today's loop dates and reads them (N44).
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5, 0.45, 0.5, 0.5]),
        env_fps=8.0,
        check_s=0.01,
        failing_reads_s=((1.2, 1.6),),
        expect=frozenset({"K-n", "U2", "X2"}),
        name="wall-a-failed-read-takes-time",
    ),
    # wall-hold-between-frames with reads and captures that take time.
    Script(
        "wall",
        2.5,
        _timed(["hold:shift:0.5", "click", "click"], [0.25, 0.25, 0.25]),
        check_s=0.01,
        capture_s=0.02,
        expect=frozenset({"X2"}),
        name="wall-a-hold-and-reads-that-take-time",
    ),
    # wall-wait-cut-by-the-end with reads of 0.01 s: the wait starts once the read before landing
    # has returned, at 0.27 s, and the check at 2 s cuts it after 1.73 s (K-c).
    Script(
        "wall",
        SHORT_S,
        _timed(["wait:2.5", "click", "click"], [0.25] * 3),
        ends_at_s=1.5,
        check_s=0.01,
        expect=frozenset({"K-b", "K-c", "K-g", "X1", "X2"}),
        name="wall-a-wait-cut-by-the-end-and-reads-take-time",
    ),
    # wall-ends-during-a-reply with reads and captures that take time: the periodic read asked at
    # 2.25 s finds the end, dated then, and the snapshot is read when it has returned (K-b, K-g).
    Script(
        "wall",
        SHORT_S,
        _timed(FORM, [0.5] * 4),
        env_fps=4.0,
        ends_at_s=2.25,
        check_s=0.01,
        capture_s=0.02,
        expect=frozenset({"K-b", "K-g", "U10", "U17"}),
        name="wall-ends-during-a-reply-and-reads-take-time",
    ),
]

# ---- 7. an operation under way at the deadline or at a frame's start ---------------------------

ACROSS = [
    # Decision 0's reply is back at 1.95 s, and the read before landing returns at 2.05 s, past
    # the deadline: today's loop leaves it not_applied_tick_limit, the kernel's deadline
    # not_applied_time_limit (K-k). Both read the snapshot once the read has returned; the kernel
    # reports the decision after it (X2).
    Script(
        "wall",
        2.0,
        _timed(["click", "click"], [1.85, 0.5]),
        check_s=0.1,
        expect=frozenset({"K-k", "X2"}),
        name="wall-the-read-before-landing-runs-past-the-deadline",
    ),
    # The same read answers exactly at the deadline, 1.875 + 0.125 s.
    Script(
        "wall",
        2.0,
        _timed(["click", "click"], [1.75, 0.5]),
        check_s=0.125,
        expect=frozenset({"K-k", "X2"}),
        name="wall-the-read-before-landing-answers-at-the-deadline",
    ),
    # The task ended at 1.92 s, and the read before landing, 1.95 to 2.05 s, finds it: the end is
    # the task's, on both sides.
    Script(
        "wall",
        2.0,
        _timed(["click", "click"], [1.85, 0.5]),
        check_s=0.1,
        ends_at_s=1.92,
        expect=frozenset({"X2"}),
        name="wall-the-read-before-landing-finds-the-end-past-the-deadline",
    ),
    # Decision 0's click executes at 1.95 s, and the read after it returns at 2.02 s: the kernel
    # took its answer, filed at 1.95 s, before the deadline and reports the decision before the
    # snapshot, as today's loop does.
    Script(
        "wall",
        2.0,
        _timed(["click", "click"], [1.81, 0.5]),
        check_s=0.07,
        expect=frozenset(),
        name="wall-the-read-after-an-action-runs-past-the-deadline",
    ),
    # Decision 0's click executes from 1.9 to 2.1 s: both read the snapshot once it has returned;
    # the kernel reports the decision after it (X2).
    Script(
        "wall",
        2.0,
        _timed(["click", "click"], [1.9, 0.5]),
        execute_s=0.2,
        expect=frozenset({"X2"}),
        name="wall-an-action-runs-past-the-deadline",
    ),
    # The same, the snapshot read taking 1.5 s: today's loop resolves the click at tick 2, where
    # it returned; the kernel once the snapshot is read, at tick 3 (PENDING).
    Script(
        "wall",
        2.0,
        _timed(["click", "click"], [1.9, 0.5]),
        execute_s=0.2,
        snapshot_s=1.5,
        expect=frozenset({"X2"}),
        name="wall-an-action-runs-past-the-deadline-and-the-snapshot-takes-time",
    ),
    # At 4 fps decision 0 lands at 1.6 s and paces to 1.75 s; decision 1's capture runs to 2.05 s,
    # past the deadline. The kernel handles the deadline once it has returned and starts no
    # generation; today's loop starts one, settles it after its reply at 2.55 s and reads the
    # snapshot then (K-a, K-g).
    Script(
        "wall",
        2.0,
        _timed(["click"] * 3, [1.3, 0.5, 0.5]),
        env_fps=4.0,
        input_fps=4.0,
        capture_s=0.3,
        expect=frozenset({"K-a", "K-g", "U17"}),
        name="wall-a-capture-runs-past-the-deadline",
    ),
    # Decision 0's click executes from 0.8 to 1.1 s, across the periodic read due at 1.0 s, which
    # the read after the action makes (turns.py TurnPolicy.on_timer).
    Script(
        "wall",
        3.0,
        _timed(["click"] * 3, [0.8, 0.3, 0.3]),
        execute_s=0.3,
        expect=frozenset({"X2"}),
        name="wall-an-action-runs-into-the-next-frame",
    ),
    # The guest refuses it: the read due at 1.0 s is made when the refusal has come back
    # (TurnPolicy._executed).
    Script(
        "wall",
        3.0,
        (
            Turn("click", attempts=(Attempt(0.8),), rejected="the guest refused the action"),
            *_timed(["click", "click"], [0.3, 0.3]),
        ),
        execute_s=0.3,
        expect=frozenset({"X2"}),
        name="wall-a-refused-action-runs-into-the-next-frame",
    ),
    # The task ends at 0.9 s, during the click: the read after it finds the end at 1.1 s, not
    # the periodic read due at 1.0 s (K-b).
    Script(
        "wall",
        3.0,
        _timed(["click"] * 3, [0.8, 0.3, 0.3]),
        execute_s=0.3,
        ends_at_s=0.9,
        expect=frozenset({"X2"}),
        name="wall-an-action-runs-into-the-next-frame-and-the-end",
    ),
    # Refused, the end is found at 1.1 s by the read made in the periodic one's place, and the
    # decision, its pacing already past 1.0 s, is reported before the snapshot.
    Script(
        "wall",
        3.0,
        (
            Turn("click", attempts=(Attempt(0.8),), rejected="the guest refused the action"),
            *_timed(["click", "click"], [0.3, 0.3]),
        ),
        execute_s=0.3,
        ends_at_s=0.9,
        expect=frozenset(),
        name="wall-a-refused-action-runs-into-the-next-frame-and-the-end",
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
    + TIMED
    + ACROSS
)
