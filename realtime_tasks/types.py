"""Shared task metadata, lifecycle contract and evaluation result types."""

from __future__ import annotations
import math
import time


class Task:
    seed = 42
    # Seed the environment reports it actually applied; None when it exposes no seeded randomness.
    applied_seed = None

    def configure_seed(self, seed: int = 42) -> None:
        from realtime_framework.entity.config import task_seed

        self.seed = task_seed(seed)

    task_id = ""
    category = ""  # "realtime" (web) | "browser_game"

    def configure_clock(
        self, mode: str = "wall", fps: float = 1.0, tokens_per_frame: int | None = None
    ) -> dict:
        """Configure the benchmark clock shared by every task.

        ``mode=token`` means that wall time must not advance task state.  Concrete
        backends extend :meth:`activate_clock` and :meth:`advance_clock` on top of this
        logical frame counter.
        ``tokens_per_frame`` is the simulated generation rate B used by the harness, not a
        per-decision response limit. ``None`` is unlimited B: output length adds no action delay;
        after applying the action to the observed state, the harness advances exactly one frame.
        """
        if mode not in {"wall", "token"}:
            raise ValueError(f"unsupported clock mode: {mode!r}")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if tokens_per_frame is not None and tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive or None")
        self._clock_config = {
            "mode": mode,
            "fps": float(fps),
            "tokens_per_frame": tokens_per_frame,
            "frame": 0,
            "active": False,
        }
        return self.clock_snapshot()

    def activate_clock(self) -> dict:
        config = getattr(self, "_clock_config", None)
        if config is None:
            self.configure_clock()
            config = self._clock_config
        config["active"] = True
        return self.clock_snapshot()

    def start_episode(self) -> dict:
        """Release the episode prepared by ``reset_episode`` and report the scored origin.

        ``reset_episode`` prepares a held episode; every framework setup step runs before this
        call. ``clock_origin`` is the Unix time at which task state began to advance (never later
        than the true start). ``start_hold`` is False when the backend could not hold the episode,
        in which case the origin is the reset instant and setup time counts against the agent.
        """
        return {"clock_origin": time.time(), "start_hold": True}

    def advance_clock(self, frames: float = 1) -> dict:
        config = getattr(self, "_clock_config", None)
        if config is None or not config.get("active"):
            raise RuntimeError("task clock must be configured and activated before advancing")
        if (
            not isinstance(frames, (int, float))
            or isinstance(frames, bool)
            or not math.isfinite(frames)
            or frames < 0
        ):
            raise ValueError("frames must be a non-negative finite number")
        config["frame"] += frames
        if abs(config["frame"] - round(config["frame"])) < 1e-9:
            config["frame"] = int(round(config["frame"]))
        return self.clock_snapshot()

    def clock_snapshot(self) -> dict:
        config = dict(
            getattr(
                self,
                "_clock_config",
                {
                    "mode": "wall",
                    "fps": 1.0,
                    "tokens_per_frame": None,
                    "frame": 0,
                    "active": False,
                },
            )
        )
        config["world_time_s"] = round(config["frame"] / config["fps"], 9)
        config["tokens_per_world_second"] = (
            None
            if config["tokens_per_frame"] is None
            else config["tokens_per_frame"] * config["fps"]
        )
        return config

    def description(self) -> str:
        raise NotImplementedError

    def provision(self):
        raise NotImplementedError  # -> url | None

    def verify(self) -> dict:
        raise NotImplementedError

    def interactable_providers(self) -> tuple:
        """Public task-owned affordance enrichments; DOM geometry remains observation-owned."""
        return ()
