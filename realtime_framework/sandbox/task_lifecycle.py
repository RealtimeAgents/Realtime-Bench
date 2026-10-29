"""What the host knows of a registered task: which of the hooks the Task base class does not
declare it has, and the order main() calls its lifecycle in.

The task runtimes (realtime_tasks/types.py Task, the games' realtime_tasks/game/runtime.py and the
web tasks' realtime_tasks/web/runtime.py) own their hooks: how a page starts, its protocol and its
score. The runner, GuestEnvironment, the two shells and the re-evaluation ask a TaskLifecycle and
never probe the task or branch on its category themselves. A hook is looked up when it is asked
for, so a task patched after the lifecycle was made is asked as it is then.
"""

from __future__ import annotations

import traceback


class EnvironmentInvalidError(RuntimeError):
    """Startup health checks failed; the result must not be attributed to the model."""


# The categories whose tasks hold their episode: reset_episode() prepares a held fresh episode and
# start_episode() releases it.
_HELD_EPISODE_CATEGORIES = frozenset({"browser_game", "realtime"})
# A web task's page renders the held, restarted state within this long; the wall clock waits it
# out before it starts.
_WEB_RENDER_PAUSE_S = 0.45


class TaskLifecycle:
    def __init__(self, task):
        self.task = task

    @property
    def category(self) -> str | None:
        """The task's category: "browser_game" for a game, "realtime" for a web task."""
        return getattr(self.task, "category", None)

    @property
    def debug_port(self) -> int | None:
        """The debugging port of the Chrome that shows the task's page."""
        return getattr(self.task, "dbg_port", None)

    @property
    def applied_seed(self):
        """The seed the task reports it applied; None when it has no seeded randomness."""
        return getattr(self.task, "applied_seed", None)

    @property
    def last_clock_advance(self) -> dict | None:
        """What the task's last token-clock advance or render reported, its render fence with it."""
        return getattr(self.task, "_last_clock_advance", None)

    @property
    def end_check(self):
        """The task's episode_over, GuestEnvironment's stop_check: the server-owned end of a web
        task, a game's terminal state or reached target. None when the task has none."""
        return getattr(self.task, "episode_over", None)

    @property
    def snapshot_reader(self):
        """The task's evaluation_snapshot, which the shells read at the freeze; None when the
        task has none (the Task base class has none)."""
        return getattr(self.task, "evaluation_snapshot", None)

    @property
    def frame_renderer(self):
        """The task's render_current_frame, which commits an action's pixels without moving the
        token clock; None when the task commits no frame."""
        renderer = getattr(self.task, "render_current_frame", None)
        return renderer if callable(renderer) else None

    @property
    def takes_clock_keys(self) -> bool:
        """Whether the task takes key input through the token clock (clock_key_event), as a
        game does."""
        return hasattr(self.task, "clock_key_event")

    @property
    def reload_discards_episode(self) -> bool:
        """Reloading a game's page restarts the game and discards the episode; a web task's
        server keeps its state across a reload."""
        return self.category == "browser_game"

    def begin_gui_action(self):
        """Before a GUI action: the page barrier's snapshot, for a task that keeps logical time
        frozen until the business request the input causes completes (a web task); None for a
        task without one."""
        begin = getattr(self.task, "begin_gui_action", None)
        complete = getattr(self.task, "complete_gui_action", None)
        return begin() if callable(begin) and callable(complete) else None

    def complete_gui_action(self, snapshot):
        """After the GUI action whose barrier ``begin_gui_action`` returned ``snapshot``."""
        return self.task.complete_gui_action(snapshot)

    def provision(self, seed: int, clock_mode: str, fps: float, tokens_per_frame) -> tuple:
        """Seed the task and configure its clock, then provision it: a game's provision reads both.
        Returns the task description and the URL main() opens with ``debug_port``, None when the
        task opened its own page."""
        self.task.configure_seed(seed)
        description = self.task.description()
        self.task.configure_clock(clock_mode, fps, tokens_per_frame)
        return description, self.task.provision()

    def prepare_first_frame(self, clock_mode: str, clock) -> None:
        """Before the agent's initialize(): a fresh held episode for the first frame, then the
        task clock activated. main() restarts a wall-clock episode once more after initialize()
        (start_scored_episode), so this warm-up does not consume its scored action window. A
        failed restart raises EnvironmentInvalidError; ``clock`` waits for a web page's render."""
        if self._holds_episode():
            category = self.category
            try:
                baseline = self.task.reset_episode()
                print(f"[{category}] fresh episode reset; baseline={baseline}", flush=True)
                if category == "realtime" and clock_mode != "token":
                    clock.sleep(_WEB_RENDER_PAUSE_S)
            except Exception:
                detail = traceback.format_exc()
                print(f"[{category}] fresh restart FAILED; run is invalid:\n" + detail, flush=True)
                raise EnvironmentInvalidError(
                    f"{category} episode restart failed before tick 0\n" + detail
                )
        self.task.activate_clock()
        print(f"[clock] {self.task.clock_snapshot()}", flush=True)

    def start_scored_episode(self, clock_mode: str, clock) -> tuple:
        """After the agent's initialize(), on the wall clock: restart the held episode in two
        phases, reset_episode() (reload, baseline, semantics, browser render) and start_episode(),
        which releases it. The origin start_episode() reports is the scored wall-clock origin, so
        no framework setup consumes the episode. Returns that origin and start_episode()'s record,
        (None, None) on the token clock and for a task that holds no episode. A failed restart
        raises EnvironmentInvalidError."""
        if clock_mode != "wall" or not self._holds_episode():
            return None, None
        category = self.category
        try:
            baseline = self.task.reset_episode()
            print(
                f"[{category}] scored wall episode reset after adapter initialization; "
                f"baseline={baseline}",
                flush=True,
            )
            if category == "realtime":
                clock.sleep(_WEB_RENDER_PAUSE_S)
            started = self.task.start_episode()
            origin = float(started["clock_origin"])
            print(f"[{category}] scored wall episode started; {started}", flush=True)
        except Exception:
            detail = traceback.format_exc()
            print(
                f"[{category}] scored wall restart FAILED; run is invalid:\n" + detail, flush=True
            )
            raise EnvironmentInvalidError(
                f"{category} episode restart failed after adapter initialization\n" + detail
            )
        return origin, started

    def _holds_episode(self) -> bool:
        return hasattr(self.task, "reset_episode") and self.category in _HELD_EPISODE_CATEGORIES

    def evaluate(self, snapshot) -> dict:
        """The task's score of a frozen snapshot, for main() and the re-evaluation of a saved
        one alike."""
        evaluate = getattr(self.task, "evaluate_snapshot", None)
        if evaluate is None:
            raise RuntimeError(f"task {self.task.task_id} cannot evaluate saved state")
        return evaluate(snapshot)
