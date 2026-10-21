"""The loops a replayed script can run under, all called the way the production shell will be:

    driver(agent, env, config, *, clocks, log, trace) -> timeline

with ``config`` the episode's TurnConfig, as main() would build it, and the timeline in
run_stream's shape (k, t_k, d_ticks, tau_k, action). ``legacy`` is today's loop; the shell's
run_episode (wave2-plan.md, P1 对外接口) takes this signature as it is.
"""

from __future__ import annotations

from realtime_framework.clock import Clocks
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.evaluation.runner import run_stream


def legacy(agent, env, config: TurnConfig, *, clocks: Clocks, log, trace) -> list:
    """run_stream with the arguments main() passes it (evaluation/runner.py:1822-1835)."""
    return run_stream(
        agent,
        env,
        delta_s=1.0 / config.env_fps,
        frame_ticks=config.env_fps / config.input_fps,
        max_ticks=config.max_ticks,
        log=log,
        think_budget_s=config.think_budget_s,
        per_move_cap_s=config.per_move_cap_s,
        clock_mode=config.clock_mode,
        tokens_per_frame=config.tokens_per_frame,
        trace=trace,
        clock=clocks.operational,
    )
