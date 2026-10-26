"""The loops a replayed script runs under, both called the way main() calls them:

    driver(agent, env, config, *, clocks, log, trace) -> EpisodeResult

with ``config`` the episode's TurnConfig and the result's timeline in the legacy loop's shape
(k, t_k, d_ticks, tau_k, action). ``legacy`` is the turn loop main() ran before the switch, kept to
run side by side (legacy_loop.py _run_stream_legacy; decision 11), and ``shell`` the kernel's
(wave2-plan.md, P1 对外接口); the environment the script's world is built on is the same for both.
"""

from __future__ import annotations

from realtime_framework.clock import Clocks
from realtime_framework.evaluation.episode_end import EpisodeEnd, EpisodeResult
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.evaluation.shell import run_episode

from .legacy_loop import _run_stream_legacy


def legacy(agent, env, config: TurnConfig, *, clocks: Clocks, log, trace) -> EpisodeResult:
    """The legacy loop with the arguments main() passed it before the switch, the horizon from the
    episode's minutes as there; the kernel takes it within 1e-9 of a whole tick as that tick
    (kernel/turns.py TurnConfig). ``agent`` takes the loop's calls (legacy_turn.py LegacyTurn).
    What the projection compares of how the episode came out is what the loop leaves on ``env``,
    where main() read it then (legacy_loop.py _finalize)."""
    timeline = _run_stream_legacy(
        agent,
        env,
        delta_s=1.0 / config.env_fps,
        frame_ticks=config.env_fps / config.input_fps,
        max_ticks=config.duration_s / 60 * 60 * config.env_fps,
        log=log,
        think_budget_s=config.think_budget_s,
        per_move_cap_s=config.per_move_cap_s,
        clock_mode=config.clock_mode,
        tokens_per_frame=config.tokens_per_frame,
        trace=trace,
        clock=clocks.operational,
    )
    end = EpisodeEnd(
        termination_record=getattr(env, "_termination", None),
        judged=True,
        invalid_run=getattr(env, "_invalid_run", None),
        loop_error=getattr(env, "_loop_error", None),
    )
    accounting = getattr(env, "_token_accounting", None)
    return end.result(env, timeline, accounting, env._drain_abandoned)


def shell(agent, env, config: TurnConfig, *, clocks: Clocks, log, trace) -> EpisodeResult:
    """run_episode (evaluation/shell.py), on the agent's kernel methods. The kernel counts the
    failed state reads, so the world's environment passes them on as production's does
    (legacy_environment.py LegacyStateReads, replayed.py WallEnvironment)."""
    env.kernel_counts_reads = True
    return run_episode(agent, env, config, clocks=clocks, log=log, trace=trace)
