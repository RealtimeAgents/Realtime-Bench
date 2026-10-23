"""The loops a replayed script runs under, both called the way main() calls them:

    driver(agent, env, config, *, clocks, log, trace) -> timeline

with ``config`` the episode's TurnConfig and the timeline in the legacy loop's shape (k, t_k,
d_ticks, tau_k, action). ``legacy`` is the turn loop main() ran before the switch, kept to run side
by side (legacy_loop.py _run_stream_legacy; decision 11), and ``shell`` the kernel's
(wave2-plan.md, P1 对外接口); the environment the script's world is built on is the same for both.
"""

from __future__ import annotations

from realtime_framework.clock import Clocks
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.evaluation.shell import run_episode

from .legacy_loop import _run_stream_legacy


def legacy(agent, env, config: TurnConfig, *, clocks: Clocks, log, trace) -> list:
    """The legacy loop with the arguments main() passed it before the switch, the horizon from the
    episode's minutes as there; the kernel takes it within 1e-9 of a whole tick as that tick
    (kernel/turns.py TurnConfig)."""
    return _run_stream_legacy(
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


def shell(agent, env, config: TurnConfig, *, clocks: Clocks, log, trace) -> list:
    """run_episode (evaluation/shell.py). Until P2 the agent's kernel methods are the defaults
    that run the legacy turn (entity/harness.py StreamingAgent.generate). The kernel counts the
    failed state reads, so the world's environment passes them on as production's does
    (legacy_environment.py LegacyStateReads, replayed.py WallEnvironment)."""
    env.kernel_counts_reads = True
    return run_episode(agent, env, config, clocks=clocks, log=log, trace=trace)
