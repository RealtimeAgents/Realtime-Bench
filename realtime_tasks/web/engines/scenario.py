from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import random
import time
import uuid
from typing import Callable


@dataclass(frozen=True)
class RecordSpec:
    label: str
    baseline: str
    signal: str
    revision: str
    platform: str | None = None
    fare_cents: int | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    family: str
    seed: int
    title: str
    subtitle: str
    policy: str
    actions: tuple[tuple[str, str], ...]
    records: tuple[RecordSpec, ...]
    target_label: str
    correct_action: str
    success_message: str
    event_delay: float
    revision_after: float = 0.25
    action_window: float = 300.0
    action_version: int = 2
    stream_interval: float = 20.0


from realtime_tasks.registry import web_definitions, load_task_module, task_directory
import json

_CONFIGURED = {
    tid: json.loads((task_directory("web", tid) / d["scenario_file"]).read_text())
    for tid, d in web_definitions().items()
    if d["backend"] == "configured"
}


def _spec(data):
    data = dict(data)
    data["records"] = tuple(RecordSpec(**record) for record in data["records"])
    data["actions"] = tuple(tuple(action) for action in data["actions"])
    return ScenarioSpec(**data)


SPECS = {tid: _spec(data["spec"]) for tid, data in _CONFIGURED.items()}
_PRESENTATIONS = {
    tid: data["presentation"]
    for tid, data in _CONFIGURED.items()
    if data["presentation"] is not None
}
_SCENE_RULES = {
    tid: data["scene_rules"] for tid, data in _CONFIGURED.items() if data["scene_rules"] is not None
}
_LIVE_EVENTS = {
    tid: data["live_events"] for tid, data in _CONFIGURED.items() if data["live_events"] is not None
}
_STRICT_COMMAND_ACCOUNTING = {
    tid for tid, data in _CONFIGURED.items() if data["strict_command_accounting"]
}
_WORLDS = {
    tid: load_task_module(tid, "environment/world.py")
    for tid in _CONFIGURED
    if web_definitions()[tid].get("world")
}
# Optional task-owned overrides for the record view, payload and action policy.
_HOOKS = {
    tid: load_task_module(tid, "environment/hooks.py")
    for tid in _CONFIGURED
    if web_definitions()[tid].get("hooks")
}
INTERACTIVE_SCENARIOS = {
    tid: provider.SCENARIOS[tid] for tid, provider in _WORLDS.items() if provider.SCENARIOS.get(tid)
}


def create_world(scenario, episode_key=None):
    return _WORLDS[scenario].create_world(scenario, episode_key) if scenario in _WORLDS else None


def public_world(scenario, world, stream_seq, elapsed_seconds):
    return _WORLDS[scenario].public_world(scenario, world, stream_seq, elapsed_seconds)


def apply_command(scenario, world, command, stream_seq, elapsed_seconds):
    return _WORLDS[scenario].apply_command(scenario, world, command, stream_seq, elapsed_seconds)


def terminal_failure_reason(scenario, world, stream_seq, elapsed_seconds):
    return _WORLDS[scenario].terminal_failure_reason(scenario, world, stream_seq, elapsed_seconds)


def automatic_resolution(scenario, world, elapsed_seconds):
    callback = getattr(_WORLDS[scenario], "automatic_resolution", None)
    return bool(callback and callback(scenario, world, elapsed_seconds))


def evaluation_details(scenario, world, stream_seq, elapsed_seconds):
    import inspect

    callback = _WORLDS[scenario].evaluation_details
    if len(inspect.signature(callback).parameters) == 3:
        return callback(scenario, world, elapsed_seconds)
    return callback(scenario, world, stream_seq, elapsed_seconds)


class RealtimeFamily:
    """Shared deterministic event/action engine used by each visual family."""

    family = "base"

    def __init__(
        self, spec: ScenarioSpec, clock: Callable[[], float], wall_clock: Callable[[], float]
    ):
        if spec.family != self.family:
            raise ValueError(f"{spec.name} uses {spec.family}, not {self.family}")
        self.spec = spec
        self.clock = clock
        self.wall_clock = wall_clock
        self.template = f"scenarios/{spec.name}.html"
        self.scene_rule = _SCENE_RULES[spec.name]
        self.hooks = _HOOKS.get(spec.name)

    def reset(self, generation: int = 1, seed: int = 42) -> dict:
        started = self.clock()
        episode_id = uuid.uuid4().hex
        records = [
            {
                "ref": f"case-{uuid.uuid4().hex[:12]}",
                "spec": record,
                "source_index": index,
                "scene": self.scene_rule["records"][index],
            }
            for index, record in enumerate(self.spec.records)
        ]
        random.Random(seed).shuffle(records)
        event_at = started + self.spec.event_delay
        revision_at = event_at + self.spec.revision_after
        action_at = revision_at if self.spec.action_version == 3 else event_at
        return {
            "scenario": self.spec.name,
            "episode": episode_id,
            "world": create_world(self.spec.name, str(seed)),
            "generation": generation,
            "view_tokens": {version: uuid.uuid4().hex for version in (1, 2, 3)},
            "seed": seed,
            "started": started,
            "wall_started": self.wall_clock(),
            "opened": False,
            "event_at": event_at,
            "revision_at": revision_at,
            "action_at": action_at,
            "deadline": action_at + self.spec.action_window,
            "records": records,
            "accounting": {"early": 0, "stale": 0, "late": 0, "wrong": 0},
            "attempts": [],
            "attempt_count": 0,
            "action": None,
            "resolved": False,
        }

    def _version(self, episode: dict, now: float) -> int:
        if now < episode["event_at"]:
            return 1
        if self.spec.action_version == 2 or now < episode["revision_at"]:
            return 2
        return 3

    def _stream_seq(self, episode: dict, now: float) -> int:
        return int(max(0.0, now - episode["started"]) // self.spec.stream_interval)

    def _response_window_closed(self, episode: dict, now: float) -> bool:
        return now > episode["deadline"]

    def _terminal_failure_reason(self, episode: dict, now: float) -> str | None:
        if episode.get("world") is not None:
            embedded = terminal_failure_reason(
                self.spec.name,
                episode["world"],
                self._stream_seq(episode, now),
                max(0.0, now - episode["started"]),
            )
            if embedded:
                return embedded
        if self._response_window_closed(episode, now):
            return "deadline_expired"
        return None

    def _wall_ms(self, episode: dict, monotonic_time: float) -> int:
        return round((episode["wall_started"] + monotonic_time - episode["started"]) * 1000)

    def _record_message(
        self, record: RecordSpec, version: int, last_event_seq: int | None, record_index: int
    ) -> str:
        if version == 1:
            evidence = record.baseline
        elif version == 2:
            evidence = f"{record.baseline} · Live signal: {record.signal}"
        else:
            evidence = f"{record.baseline} · Live signal: {record.signal} · Latest confirmation: {record.revision}"
        if last_event_seq is None:
            return evidence
        events = _LIVE_EVENTS[self.spec.name]
        event_round = last_event_seq // len(self.spec.records)
        activity = events[(event_round + record_index) % len(events)].format(label=record.label)
        return f"{evidence} · Live update {last_event_seq + 1}: {activity}"

    def family_metrics(self, episode: dict, now: float) -> list[dict]:
        stream_seq = self._stream_seq(episode, now)
        return [
            {"label": "Live records", "value": str(len(episode["records"]))},
            {"label": "Stream update", "value": f"#{stream_seq + 1}"},
            {
                "label": "Decision state",
                "value": "Complete" if episode["resolved"] else "Monitoring",
            },
        ]

    def state(self, episode: dict) -> dict:
        now = self.clock()
        version = self._version(episode, now)
        stream_seq = self._stream_seq(episode, now)
        if (
            not episode["resolved"]
            and episode.get("world") is not None
            and automatic_resolution(
                self.spec.name, episode["world"], max(0.0, now - episode["started"])
            )
        ):
            episode["resolved"] = not self._has_fatal(episode)
            if episode["resolved"]:
                episode["action"] = {
                    "record": "workspace",
                    "action": "wait",
                    "at": now,
                    "version": version,
                    "generation": episode["generation"],
                }
        terminal_reason = (
            None if episode["resolved"] else self._terminal_failure_reason(episode, now)
        )
        if episode["resolved"]:
            phase = "resolved"
        elif now < episode["event_at"]:
            phase = "watching"
        elif version < self.spec.action_version:
            phase = "confirming"
        elif terminal_reason:
            phase = "expired"
        else:
            phase = "active"
        records = []
        record_count = len(episode["records"])
        for record_index, item in enumerate(episode["records"]):
            scene = item["scene"]
            last_event_seq = (
                stream_seq - (stream_seq - record_index) % record_count
                if stream_seq >= record_index
                else None
            )
            record = {
                "ref": item["ref"],
                "label": item["spec"].label,
                "message": self._record_message(
                    item["spec"], version, last_event_seq, record_index
                ),
                "updated": version > 1,
                "facts": [
                    {
                        "label": label,
                        "value": value
                        if version >= self.spec.action_version
                        else "Awaiting confirmation"
                        if version > 1
                        else "Pending live signal",
                    }
                    for label, value in scene["facts"]
                ],
            }
            if self.hooks and hasattr(self.hooks, "record_view"):
                self.hooks.record_view(self, episode, item, record, version, now, stream_seq)
            record["arrival_index"] = item["source_index"]
            records.append(record)
        public_deadline_ms = self._wall_ms(episode, episode["deadline"])
        payload = {
            "ok": True,
            "episode": episode["episode"],
            "generation": episode["generation"],
            "server_now_ms": self._wall_ms(episode, now),
            "phase": phase,
            "terminal_reason": terminal_reason,
            "can_act": phase == "active",
            "stream_seq": stream_seq,
            "updated_at_ms": self._wall_ms(episode, now),
            "version": version,
            "view_token": episode["view_tokens"][version],
            "event_at_ms": self._wall_ms(episode, episode["event_at"]),
            "revision_at_ms": self._wall_ms(episode, episode["revision_at"]),
            "deadline_ms": public_deadline_ms,
            "records": [] if episode.get("world") is not None else records,
            "interaction": public_world(
                self.spec.name, episode["world"], stream_seq, max(0.0, now - episode["started"])
            )
            if episode.get("world") is not None
            else None,
            "actions": []
            if episode.get("world") is not None
            else [
                {
                    "key": key,
                    "label": self.scene_rule["action_label"]
                    if key == self.spec.correct_action
                    else label,
                }
                for key, label in self.spec.actions
            ],
            "metrics": self.family_metrics(episode, now),
            "resolved": episode["resolved"],
            "result_message": self.spec.success_message if episode["resolved"] else None,
        }
        if self.hooks and hasattr(self.hooks, "payload_view"):
            self.hooks.payload_view(self, episode, payload, now)
        return payload

    _FATAL_ACCOUNTING = ("early", "late", "wrong")

    def _has_fatal(self, episode: dict) -> bool:
        return any((episode["accounting"][kind] for kind in self._FATAL_ACCOUNTING))

    def _record_attempt(self, episode: dict, attempt: dict) -> None:
        episode["attempt_count"] += 1
        if len(episode["attempts"]) < 64:
            episode["attempts"].append(attempt)

    def _commit_action(
        self, episode: dict, record_ref: str, action_key: str, now: float, version: int
    ) -> tuple[dict, int] | None:
        target = next(
            (item for item in episode["records"] if item["spec"].label == self.spec.target_label)
        )
        selected = next((item for item in episode["records"] if item["ref"] == record_ref), None)
        policy_allows = bool(selected and all(selected["scene"]["checks"]))
        if self.hooks and hasattr(self.hooks, "policy_allows") and selected is not None:
            policy_allows = self.hooks.policy_allows(self, episode, selected, now)
        if (
            record_ref != target["ref"]
            or action_key != self.spec.correct_action
            or (not policy_allows)
        ):
            return None
        episode["action"] = {
            "record": record_ref,
            "action": action_key,
            "at": now,
            "version": version,
            "generation": episode["generation"],
        }
        self._record_attempt(episode, dict(episode["action"], kind="correct"))
        episode["resolved"] = not self._has_fatal(episode)
        if episode["resolved"]:
            return ({"ok": True, "resolved": True, "message": self.spec.success_message}, 200)
        return (
            {
                "ok": False,
                "resolved": False,
                "error": "The correct response arrived, but an earlier invalid action already made this run unsuccessful.",
            },
            409,
        )

    def _command_action(
        self, episode: dict, body: dict, now: float | None = None
    ) -> tuple[dict, int]:
        """Apply one non-terminal workspace adjustment.

        Scheduling and control pages may submit many commands. The browser does not
        pre-validate their operational quality; each accepted command changes the
        server-owned world and the objective is evaluated from the resulting state.
        """
        supplied_version = body.get("version")
        supplied_generation = body.get("generation")
        view_token = body.get("view_token")
        command = body.get("command")
        supplied_command_id = body.get("command_id")
        if (
            type(supplied_version) is not int
            or type(supplied_generation) is not int
            or (not isinstance(view_token, str))
            or (len(view_token) != 32)
            or (not isinstance(command, dict))
            or (not isinstance(command.get("operation"), str))
            or (not command["operation"])
            or (len(command["operation"]) > 128)
        ):
            return (
                {
                    "ok": False,
                    "kind": "invalid",
                    "error": "Command requires a current view token, version and generation.",
                },
                400,
            )
        now = self.clock() if now is None else now
        version = self._version(episode, now)
        if supplied_generation != episode["generation"]:
            return (
                {
                    "ok": False,
                    "kind": "generation",
                    "error": "This command belongs to an earlier task restart.",
                },
                409,
            )
        if episode["resolved"]:
            return ({"ok": True, "resolved": True, "message": self.spec.success_message}, 200)
        terminal_reason = self._terminal_failure_reason(episode, now)
        if terminal_reason:
            return (
                {
                    "ok": False,
                    "kind": terminal_reason,
                    "terminal_reason": terminal_reason,
                    "error": "This task is no longer actionable.",
                },
                409,
            )
        if (
            view_token != episode["view_tokens"].get(supplied_version)
            or supplied_version != version
        ):
            episode["accounting"]["stale"] += 1
            self._record_attempt(
                episode,
                {
                    "kind": "stale",
                    "command": command,
                    "at": now,
                    "version": supplied_version,
                    "generation": supplied_generation,
                },
            )
            return (
                {
                    "ok": False,
                    "kind": "stale",
                    "error": "Live data changed; review the refreshed workspace and try again.",
                },
                409,
            )
        if now < episode["event_at"] or version < self.spec.action_version:
            episode["accounting"]["early"] += 1
            self._record_attempt(
                episode,
                {
                    "kind": "early",
                    "command": command,
                    "at": now,
                    "version": supplied_version,
                    "generation": supplied_generation,
                },
            )
            return (
                {
                    "ok": False,
                    "kind": "early",
                    "error": "The live workspace is not ready for operator commands.",
                },
                409,
            )
        stream_seq = self._stream_seq(episode, now)
        elapsed_seconds = max(0.0, now - episode["started"])
        domain_command = deepcopy(command)
        if isinstance(supplied_command_id, str) and supplied_command_id:
            domain_command["_command_id"] = supplied_command_id
        result = apply_command(
            self.spec.name, episode["world"], domain_command, stream_seq, elapsed_seconds
        )
        if not result.get("ok"):
            if self.spec.name in _STRICT_COMMAND_ACCOUNTING:
                episode["accounting"]["wrong"] += 1
                self._record_attempt(
                    episode,
                    {
                        "kind": "wrong",
                        "command": command,
                        "at": now,
                        "version": version,
                        "generation": supplied_generation,
                    },
                )
            return ({**result, "kind": "invalid"}, 400)
        deadline_elapsed = result.pop("_deadline_elapsed", None)
        if type(deadline_elapsed) in (int, float):
            episode["deadline"] = max(
                episode["deadline"], episode["started"] + float(deadline_elapsed)
            )
        self._record_attempt(
            episode,
            {
                "kind": "command",
                "command": command,
                "at": now,
                "version": version,
                "generation": supplied_generation,
            },
        )
        if result.get("resolved"):
            episode["resolved"] = not self._has_fatal(episode)
            episode["action"] = {
                "record": "workspace",
                "action": "multi_step",
                "at": now,
                "version": version,
                "generation": supplied_generation,
            }
            result["resolved"] = episode["resolved"]
            result["message"] = (
                self.spec.success_message
                if episode["resolved"]
                else "The workspace objective was reached, but an earlier invalid command made this run unsuccessful."
            )
        return (result, 200)

    def action(self, episode: dict, body: dict, now: float | None = None) -> tuple[dict, int]:
        if not isinstance(body, dict):
            return (
                {"ok": False, "kind": "invalid", "error": "Action payload must be a JSON object."},
                400,
            )
        if self.spec.name in INTERACTIVE_SCENARIOS:
            if "command" not in body:
                return (
                    {
                        "ok": False,
                        "kind": "invalid",
                        "error": "This workspace accepts multi-step commands, not record decisions.",
                    },
                    400,
                )
            return self._command_action(episode, body, now)
        supplied_version = body.get("version")
        supplied_generation = body.get("generation")
        record_ref = body.get("record")
        action_key = body.get("action")
        view_token = body.get("view_token")
        if (
            type(supplied_version) is not int
            or type(supplied_generation) is not int
            or (not isinstance(record_ref, str))
            or (not record_ref)
            or (len(record_ref) > 128)
            or (not isinstance(action_key, str))
            or (not action_key)
            or (len(action_key) > 128)
            or (not isinstance(view_token, str))
            or (len(view_token) != 32)
        ):
            return (
                {
                    "ok": False,
                    "kind": "invalid",
                    "error": "Action requires a current view token and valid record/action/version/generation fields.",
                },
                400,
            )
        now = self.clock() if now is None else now
        version = self._version(episode, now)
        if supplied_generation != episode["generation"]:
            return (
                {
                    "ok": False,
                    "kind": "generation",
                    "error": "This action belongs to an earlier task restart. Review the current state.",
                },
                409,
            )
        accepted = episode.get("action")
        if episode["resolved"] and accepted:
            replay = (
                record_ref == accepted["record"]
                and action_key == accepted["action"]
                and (supplied_version == accepted["version"])
            )
            if replay:
                return ({"ok": True, "resolved": True, "message": self.spec.success_message}, 200)
            return (
                {
                    "ok": False,
                    "kind": "resolved",
                    "error": "This episode already has an accepted decision.",
                },
                409,
            )
        terminal_reason = self._terminal_failure_reason(episode, now)
        if terminal_reason:
            return (
                {
                    "ok": False,
                    "kind": terminal_reason,
                    "terminal_reason": terminal_reason,
                    "error": "This task is no longer actionable.",
                },
                409,
            )
        if view_token != episode["view_tokens"].get(supplied_version):
            episode["accounting"]["stale"] += 1
            self._record_attempt(
                episode,
                {
                    "kind": "stale",
                    "record": record_ref,
                    "action": action_key,
                    "at": now,
                    "version": supplied_version,
                    "generation": supplied_generation,
                },
            )
            return (
                {
                    "ok": False,
                    "kind": "stale",
                    "error": "The submitted view was not issued for this record revision.",
                },
                409,
            )
        if now < episode["event_at"]:
            kind, message = ("early", "The live decision signal has not opened yet.")
        elif version < self.spec.action_version:
            kind, message = ("early", "Required confirmation evidence has not arrived yet.")
        elif supplied_version != version:
            kind, message = (
                "stale",
                "The record changed after this view was rendered. Review the current state.",
            )
        else:
            committed = self._commit_action(episode, record_ref, action_key, now, version)
            if committed is not None:
                return committed
            kind, message = (
                "wrong",
                "That record and response do not satisfy the operating policy.",
            )
        episode["accounting"][kind] += 1
        self._record_attempt(
            episode,
            {
                "kind": kind,
                "record": record_ref,
                "action": action_key,
                "at": now,
                "version": supplied_version,
                "generation": supplied_generation,
            },
        )
        return ({"ok": False, "kind": kind, "error": message}, 409)

    def evaluation(self, episode: dict) -> dict:
        from realtime_tasks.web.verifier import evaluate_engine

        return evaluate_engine(self, episode)

    def ui_context(self) -> dict:
        return {
            "title": self.spec.title,
            "subtitle": self.spec.subtitle,
            "policy": "Live operational workspace · raw source data",
            "family": self.spec.family,
            "scenario": self.spec.name,
            "primary_action": self.spec.correct_action,
            "action_label": self.scene_rule["action_label"],
            "check_labels": (),
            **_PRESENTATIONS[self.spec.name],
        }


class CommerceFamily(RealtimeFamily):
    family = "commerce"

    def family_metrics(self, episode: dict, now: float) -> list[dict]:
        return [
            {"label": "Open options", "value": str(len(episode["records"]))},
            {"label": "Queue updates", "value": str(120 + self._stream_seq(episode, now))},
            {"label": "Decision", "value": "Committed" if episode["resolved"] else "Pending"},
        ]


class OperationsFamily(RealtimeFamily):
    family = "operations"

    def family_metrics(self, episode: dict, now: float) -> list[dict]:
        return [
            {"label": "Services observed", "value": str(len(episode["records"]))},
            {"label": "Events / min", "value": str(840 + self._stream_seq(episode, now))},
            {"label": "Control state", "value": "Resolved" if episode["resolved"] else "Live"},
        ]


class CoordinationFamily(RealtimeFamily):
    family = "coordination"

    def family_metrics(self, episode: dict, now: float) -> list[dict]:
        return [
            {"label": "Resources tracked", "value": str(len(episode["records"]))},
            {"label": "Dispatch cycle", "value": str(1 + self._stream_seq(episode, now))},
            {"label": "Assignment", "value": "Confirmed" if episode["resolved"] else "Unassigned"},
        ]


FAMILY_TYPES = {
    "commerce": CommerceFamily,
    "operations": OperationsFamily,
    "coordination": CoordinationFamily,
}


def _validate_specs() -> None:
    if set(_PRESENTATIONS) != set(SPECS):
        raise RuntimeError("every realtime scenario needs one product presentation")
    if set(_SCENE_RULES) != set(SPECS):
        raise RuntimeError("every realtime scenario needs one native UI rule set")
    for spec in SPECS.values():
        scene = _SCENE_RULES[spec.name]
        if len(scene["records"]) != len(spec.records):
            raise RuntimeError(f"{spec.name}: UI record metadata is incomplete")
        if not scene["check_labels"]:
            raise RuntimeError(f"{spec.name}: UI policy checks are required")
        for row in scene["records"]:
            if len(row["checks"]) != len(scene["check_labels"]):
                raise RuntimeError(f"{spec.name}: UI check values do not match labels")
        eligible = [index for index, row in enumerate(scene["records"]) if all(row["checks"])]
        target_index = next(
            (
                index
                for index, record in enumerate(spec.records)
                if record.label == spec.target_label
            )
        )
        if spec.name not in INTERACTIVE_SCENARIOS and eligible != [target_index]:
            raise RuntimeError(f"{spec.name}: exactly the evaluator target must satisfy UI policy")
        if spec.family not in FAMILY_TYPES:
            raise RuntimeError(f"{spec.name}: unknown family {spec.family!r}")
        labels = [record.label for record in spec.records]
        if labels.count(spec.target_label) != 1:
            raise RuntimeError(f"{spec.name}: target label must identify exactly one record")
        action_keys = [key for key, _ in spec.actions]
        if len(action_keys) != len(set(action_keys)):
            raise RuntimeError(f"{spec.name}: duplicate action key")
        if spec.correct_action not in action_keys:
            raise RuntimeError(f"{spec.name}: correct action is not exposed")


def build_engines(
    clock: Callable[[], float], wall_clock: Callable[[], float] = time.time
) -> dict[str, RealtimeFamily]:
    return {
        name: FAMILY_TYPES[spec.family](spec, clock, wall_clock) for name, spec in SPECS.items()
    }


_validate_specs()
