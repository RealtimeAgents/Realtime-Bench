from __future__ import annotations

# Vision
"""Shared lifecycle, logging, and action parsing for the maintained vision agents."""

import json as json
import re as re
import threading as threading
from pathlib import Path as Path

from realtime_framework.entity.harness import StreamingAgent
from realtime_framework.harnesses.default.prompts import gui_system_prompt
from realtime_framework.harnesses.default.parsing import _literal_object
from realtime_framework.harnesses.default.parsing import unique_object
from realtime_framework.entity.actions import Action
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.usage import normalize_provider_token_usage
from realtime_framework.entity.actions import ActionTranslator

_KNOWN_ACTION_NAMES = frozenset(
    {
        "click",
        "left_click",
        "right_click",
        "double_click",
        "triple_click",
        "mouse_down",
        "mouse_up",
        "type",
        "key",
        "press",
        "hotkey",
        "keypress",
        "hold_key",
        "scroll",
        "drag",
        "move",
        "mouse_move",
        "wait",
        "done",
        "fail",
        "give_up",
    }
)


class VisionAgentBase(StreamingAgent):
    """Non-transport base of the default harness: trajectory logging and action parsing."""

    def __init__(self, config: ModelConfig, log_dir: Path | None = None, *, trace):
        self.config = config
        self.log_dir = log_dir
        # The runner's episode-trace write entry; transports and the runtime record into it.
        self._trace = trace
        self._trajectory_file = None
        self._log_lock = threading.Lock()
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._trajectory_file = open(log_dir / "trajectory.jsonl", "w")  # noqa: SIM115

        self.model_name = config.model_name
        self._setup_backend()
        self.task_instruction = ""
        self.streaming_note: str | None = None
        self.tool_runtime = None

    def bind_tool_runtime(self, runtime):
        self.tool_runtime = runtime

    def _protocol_token_fields(self, usage) -> dict:
        """Expose provider/native usage without estimating or rewriting model output."""
        capabilities = getattr(getattr(self, "driver", None), "capabilities", None)
        protocol_family = getattr(capabilities, "protocol_family", None)
        normalized = normalize_provider_token_usage(
            usage,
            protocol_family=protocol_family,
            upstream_provider=getattr(self.config, "upstream_provider", None),
        )
        return {
            **normalized,
            "provider_usage_missing_reason": (
                None if usage is not None else "backend_did_not_return_usage"
            ),
            "backend_route": None,
        }

    def initialize(self, task_instruction: str, initial_observation: Observation):
        del initial_observation
        self.task_instruction = task_instruction
        self._tok = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
            "responses": 0,
        }

    @staticmethod
    def _usage_number(usage, *names) -> int:
        if not usage:
            return 0
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return 0

    def _accumulate_usage(self, usage) -> None:
        if not usage:
            return
        inp = self._usage_number(usage, "input_tokens", "prompt_tokens")
        out = self._usage_number(usage, "output_tokens", "completion_tokens")
        total = self._usage_number(usage, "total_tokens") or inp + out
        details, output_details = {}, {}
        if isinstance(usage, dict):
            details = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
            output_details = (
                usage.get("output_token_details") or usage.get("completion_tokens_details") or {}
            )
        self._tok["input_tokens"] += inp
        self._tok["output_tokens"] += out
        self._tok["total_tokens"] += total
        self._tok["cached_input_tokens"] += int(details.get("cached_tokens", 0) or 0)
        self._tok["reasoning_output_tokens"] += int(output_details.get("reasoning_tokens", 0) or 0)
        self._tok["responses"] += 1

    @property
    def token_usage(self) -> dict:
        return dict(self._tok)

    def _log_entry(self, entry: dict):
        if not self._trajectory_file:
            return
        if entry.get("type") in {"decision", "decision_discarded"}:
            from realtime_framework.evaluation.reporting import trajectory_usage

            entry = {**entry, "usage": trajectory_usage(entry)}
            entry.pop("_token_accounting", None)
        with self._log_lock:
            self._trajectory_file.write(json.dumps(entry) + "\n")
            self._trajectory_file.flush()

    def finalize_decision_audit(self, result: dict) -> None:
        """Write a deferred trajectory row after the harness finalizes decision accounting."""
        entry = result.pop("_trajectory_entry", None)
        if not isinstance(entry, dict):
            entry = {
                "type": "decision",
                "tick": result.get("decision"),
                "decision": result.get("decision"),
                "latency_s": result.get("latency"),
                "thinking": result.get("thinking", ""),
                "model_output": result.get("raw", ""),
                "parsed_action": None,
                "error": result.get("invalid_detail"),
                "usage": result.get("usage"),
            }
            # The shell's record of a call given up on at the episode's end, which the token
            # totals and the model timing leave out (evaluation/clock.py Settlement, N34).
            if result.get("harness_abort_reason"):
                entry["harness_abort_reason"] = result["harness_abort_reason"]
        finalized_fields = (
            "decision",
            "observation_id",
            "observation_tick",
            "clock_output_tokens",
            "clock_output_tokens_source",
            "remaining_clock_tokens",
            "unclamped_clock_output_tokens",
            "post_horizon_output_tokens",
            "provider_response_status",
            "provider_response_status_reason",
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_output_tokens_source",
            "provider_reasoning_output_tokens",
            "provider_reasoning_output_tokens_source",
            "provider_visible_output_tokens",
            "provider_visible_output_tokens_source",
            "provider_visible_output_tokens_missing_reason",
            "provider_audio_output_tokens",
            "provider_audio_output_tokens_source",
            "provider_audio_output_tokens_missing_reason",
            "provider_token_normalization_schema",
            "provider_token_semantics",
            "decision_wall_latency_s",
            "provider_call_wall_latency_s",
            "runtime_turn_wall_latency_s",
            "provider_ttft_s",
            "provider_ttft_source",
            "provider_ttft_missing_reason",
        )
        entry.update({name: result.get(name) for name in finalized_fields})
        entry["_token_accounting"] = result.get("token_accounting")
        self._log_entry(entry)

    def close(self):
        if self._trajectory_file:
            self._trajectory_file.close()
        if self.tool_runtime is not None:
            self.tool_runtime.close()

    @staticmethod
    def _load_action_object(value: str) -> dict | None:
        try:
            parsed = json.loads(value, object_pairs_hook=unique_object)
        except (ValueError, RecursionError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _normalized_action_name(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        value = re.split(r"[./]", value.strip())[-1]
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
        name = re.sub(r"[\s-]+", "_", name).lower()
        aliases = {
            "tap": "click",
            "click_at": "click",
            "leftclick": "left_click",
            "rightclick": "right_click",
            "doubleclick": "double_click",
            "tripleclick": "triple_click",
            "mousedown": "mouse_down",
            "mouseup": "mouse_up",
            "mousemove": "mouse_move",
            "press_key": "key",
            "key_press": "key",
            "send_keys": "key",
            "type_text": "type",
            "input_text": "type",
            "typewrite": "type",
            "write": "type",
            "move_to": "move",
            "holdkey": "hold_key",
            "press_and_hold": "hold_key",
            "finish": "done",
            "finished": "done",
            "complete": "done",
            "completed": "done",
        }
        return aliases.get(name, name)

    @staticmethod
    def _normalized_field_name(value: object) -> str:
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value).strip())
        return re.sub(r"[\s-]+", "_", name).lower()

    @classmethod
    def _normalize_action_key_shorthand(cls, data: dict) -> dict | None:
        """Expand shorthand ``{"click": "P1"}`` action objects.

        Some text checkpoints follow the one-action constraint but serialize the verb as the
        object's key instead of putting it in an ``action`` field.  Accept that form only when
        exactly one known verb key is present, then translate its value conservatively into the
        canonical field for that verb.  Keeping this at the adapter boundary leaves the core
        action and executor contracts unchanged.
        """
        lowered = {cls._normalized_field_name(key): value for key, value in data.items()}
        if (
            cls._normalized_action_name(lowered.get("type")) in _KNOWN_ACTION_NAMES
            and len(lowered) > 1
        ):
            return data
        selector_keys = {
            "action",
            "action_name",
            "action_type",
            "name",
            "tool",
            "tool_name",
            "command",
            "operation",
            "function",
            "function_call",
            "tool_call",
        }
        if set(lowered) & selector_keys:
            return data

        action_keys = [
            key for key in lowered if cls._normalized_action_name(key) in _KNOWN_ACTION_NAMES
        ]
        if len(action_keys) != 1:
            return data

        action_key = action_keys[0]
        action_name = cls._normalized_action_name(action_key)
        payload = lowered.pop(action_key)
        parameters: dict = {}

        if isinstance(payload, dict):
            parameters = {cls._normalized_field_name(key): value for key, value in payload.items()}
        elif action_name in {
            "click",
            "left_click",
            "right_click",
            "double_click",
            "triple_click",
            "mouse_down",
            "mouse_up",
            "move",
            "mouse_move",
        }:
            if isinstance(payload, str):
                parameters["target"] = payload
            elif isinstance(payload, (list, tuple)) and len(payload) == 2:
                parameters["coordinate"] = list(payload)
            elif payload is not None:
                return data
        elif action_name == "type":
            if not isinstance(payload, str):
                return data
            parameters["text"] = payload
        elif action_name in {"key", "press", "hotkey", "keypress", "hold_key"}:
            if isinstance(payload, str):
                parameters["key"] = payload
            elif isinstance(payload, (list, tuple)):
                parameters["keys"] = list(payload)
            else:
                return data
        elif action_name == "scroll":
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                parameters.update({"dx": payload[0], "dy": payload[1]})
            elif payload is not None:
                return data
        elif action_name == "drag":
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                parameters.update({"start": payload[0], "end": payload[1]})
            elif payload is not None:
                return data
        elif action_name == "wait":
            if isinstance(payload, (int, float)) and not isinstance(payload, bool):
                parameters["duration"] = payload
            elif payload not in (None, True):
                return data
        elif action_name in {"done", "fail", "give_up"}:
            if payload not in (None, True):
                return data
        elif payload is not None:
            return data

        if set(parameters) & set(lowered):
            return data
        return {"action": action_name, **lowered, **parameters}

    @classmethod
    def _normalize_action_object(cls, data: dict) -> dict | None:
        """Flatten common JSON/tool envelopes into the canonical action object."""
        data = cls._normalize_action_key_shorthand(data)
        if data is None:
            return None
        lowered = {cls._normalized_field_name(key): value for key, value in data.items()}
        if len(lowered) != len(data):
            return None
        if (
            "type" in lowered
            and cls._normalized_action_name(lowered["type"]) in _KNOWN_ACTION_NAMES
        ):
            if "action" in lowered and cls._normalized_action_name(
                lowered["action"]
            ) != cls._normalized_action_name(lowered["type"]):
                return None
            lowered.setdefault("action", lowered.pop("type"))
        selectors = [
            lowered[k]
            for k in (
                "action",
                "action_name",
                "action_type",
                "name",
                "tool",
                "tool_name",
                "command",
                "operation",
            )
            if isinstance(lowered.get(k), str)
        ]
        if len({cls._normalized_action_name(value) for value in selectors}) > 1:
            return None
        nested = None
        if (
            sum(
                isinstance(lowered.get(key), dict)
                for key in ("function_call", "function", "tool_call")
            )
            > 1
        ):
            return None
        for key in ("function_call", "function", "tool_call"):
            if isinstance(lowered.get(key), dict):
                nested = lowered[key]
                break
        action_value = next(
            (
                lowered[key]
                for key in (
                    "action",
                    "action_name",
                    "action_type",
                    "name",
                    "tool",
                    "tool_name",
                    "command",
                    "operation",
                )
                if key in lowered
            ),
            None,
        )
        if isinstance(action_value, dict):
            nested = action_value
            action_value = None
        top_action_name = cls._normalized_action_name(action_value)
        if nested is not None:
            nested_lowered = {
                cls._normalized_field_name(key): value for key, value in nested.items()
            }
            nested_selectors = [
                nested_lowered[key]
                for key in (
                    "action",
                    "action_name",
                    "action_type",
                    "name",
                    "tool",
                    "tool_name",
                    "command",
                    "operation",
                )
                if isinstance(nested_lowered.get(key), str)
            ]
            if len({cls._normalized_action_name(value) for value in nested_selectors}) > 1:
                return None
            nested_action_value = next(
                (
                    nested_lowered[key]
                    for key in (
                        "action",
                        "action_name",
                        "action_type",
                        "name",
                        "tool",
                        "tool_name",
                        "command",
                        "operation",
                    )
                    if key in nested_lowered
                ),
                None,
            )
            nested_action_name = cls._normalized_action_name(nested_action_value)
            if (
                top_action_name in _KNOWN_ACTION_NAMES
                and nested_action_name in _KNOWN_ACTION_NAMES
                and top_action_name != nested_action_name
            ):
                return None
            action_name = (
                nested_action_name
                if nested_action_name in _KNOWN_ACTION_NAMES
                else top_action_name or nested_action_name
            )
        else:
            nested_lowered = {}
            action_name = top_action_name

        if action_name is None:
            return None

        reserved = {
            "action",
            "action_name",
            "action_type",
            "name",
            "tool",
            "tool_name",
            "command",
            "operation",
            "type",
            "id",
            "call_id",
            "function",
            "function_call",
            "tool_call",
            "arguments",
            "parameters",
            "params",
            "args",
            "input",
            "tool_input",
        }
        parameters = {key: value for key, value in lowered.items() if key not in reserved}
        nested_parameters = {
            key: value for key, value in nested_lowered.items() if key not in reserved
        }
        if any(
            key in parameters and parameters[key] != value
            for key, value in nested_parameters.items()
        ):
            return None
        parameters.update(nested_parameters)
        containers = [
            source[key]
            for source in (nested_lowered, lowered)
            for key in ("arguments", "parameters", "params", "args", "input", "tool_input")
            if key in source
        ]
        if len(containers) > 1:
            return None
        parameter_container = next(
            (
                source[key]
                for source in (nested_lowered, lowered)
                for key in (
                    "arguments",
                    "parameters",
                    "params",
                    "args",
                    "input",
                    "tool_input",
                )
                if key in source
            ),
            None,
        )
        if isinstance(parameter_container, str):
            try:
                parameter_container = json.loads(
                    parameter_container, object_pairs_hook=unique_object
                )
            except (TypeError, ValueError):
                try:
                    parameter_container = _literal_object(parameter_container)
                except (SyntaxError, ValueError, TypeError, RecursionError):
                    return None
        if parameter_container is not None:
            if not isinstance(parameter_container, dict):
                return None
            supplied = {
                cls._normalized_field_name(key): value for key, value in parameter_container.items()
            }
            if any(
                key in parameters and parameters[key] != value for key, value in supplied.items()
            ):
                return None
            parameters.update(supplied)
        if "action" in parameters:
            inner_name = cls._normalized_action_name(parameters.pop("action"))
            if action_name in _KNOWN_ACTION_NAMES and inner_name != action_name:
                return None
            action_name = inner_name

        field_aliases = {
            "coordinates": "coordinate",
            "coords": "coordinate",
            "position": "coordinate",
            "location": "coordinate",
            "point": "coordinate",
            "target_id": "target",
            "reference": "target",
            "ref": "target",
            "duration_seconds": "duration",
            "seconds": "duration",
            "text_to_type": "text",
            "delta_x": "dx",
            "delta_y": "dy",
        }
        normalized_parameters = {}
        for key, value in parameters.items():
            key = field_aliases.get(key, key)
            if key in normalized_parameters and normalized_parameters[key] != value:
                return None
            normalized_parameters[key] = value
        parameters = normalized_parameters
        duration_ms = parameters.pop("duration_ms", None)
        if "duration" not in parameters and isinstance(duration_ms, (int, float, str)):
            try:
                parameters["duration"] = float(duration_ms) / 1000
            except ValueError:
                parameters["duration"] = duration_ms
        if action_name == "type" and "text" not in parameters and "value" in parameters:
            parameters["text"] = parameters.pop("value")
        if isinstance(parameters.get("keys"), str) and "key" not in parameters:
            parameters["key"] = parameters.pop("keys")
        coordinate = parameters.get("coordinate")
        if isinstance(coordinate, dict):
            coordinate = {
                cls._normalized_field_name(key): value for key, value in coordinate.items()
            }
            if "x" in coordinate and "y" in coordinate:
                parameters.pop("coordinate")
                parameters.setdefault("x", coordinate["x"])
                parameters.setdefault("y", coordinate["y"])
        if action_name == "drag":
            for field, x_field, y_field in (
                ("start", "x1", "y1"),
                ("end", "x2", "y2"),
            ):
                point = parameters.get(field)
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    parameters.pop(field)
                    parameters.setdefault(x_field, point[0])
                    parameters.setdefault(y_field, point[1])
                elif isinstance(point, dict):
                    point = {cls._normalized_field_name(key): value for key, value in point.items()}
                    if "x" in point and "y" in point:
                        parameters.pop(field)
                        parameters.setdefault(x_field, point["x"])
                        parameters.setdefault(y_field, point["y"])

        for key in (
            "x",
            "y",
            "x1",
            "y1",
            "x2",
            "y2",
            "dx",
            "dy",
            "duration",
            "repeat",
        ):
            value = parameters.get(key)
            if not isinstance(value, str):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            parameters[key] = int(number) if number.is_integer() else number
        for key in ("button", "modifier"):
            if isinstance(parameters.get(key), str):
                parameters[key] = parameters[key].strip().lower()
        return {"action": action_name, **parameters}

    def _action_object(self, content: str) -> dict | None:
        if not content:
            return None
        loaded = self._load_action_object(content)
        data = self._normalize_action_object(loaded) if loaded is not None else None
        if data is None or data.get("action") not in _KNOWN_ACTION_NAMES:
            return None
        return data

    def _parse_action(self, content: str) -> Action | None:
        data = self._action_object(content)
        return None if data is None else ActionTranslator.from_json(data)

    def _parse_framed_action(self, content: str) -> tuple[Action | None, object]:
        """A streaming action, and the value of its frame field: the frame it is based on."""
        data = self._action_object(content)
        if data is None:
            return None, None
        reference = data.pop("frame", None)
        return ActionTranslator.from_json(data), reference


# Profiles
"""Model-behavior profiles independent of provider wire protocols."""

from dataclasses import dataclass as dataclass

from realtime_framework.harnesses.default.prompts import FINITE_BUDGET_FUNCTION_CALL_OUTPUT_CONTRACT
from realtime_framework.harnesses.default.prompts import FUNCTION_CALL_ACTION_SPACE
from realtime_framework.harnesses.default.prompts import FUNCTION_CALL_FRAME_PROMPT
from realtime_framework.harnesses.default.prompts import FUNCTION_CALL_FRAME_REFERENCE_RULES
from realtime_framework.harnesses.default.prompts import FUNCTION_CALL_OUTPUT_CONTRACT
from realtime_framework.harnesses.default.prompts import FRAME_REFERENCE_RULES
from realtime_framework.harnesses.default.prompts import GROUNDING_FAILED_FEEDBACK
from realtime_framework.harnesses.default.prompts import SCENARIO_ROLES
from realtime_framework.harnesses.default.prompts import STREAM_OUTPUT_CONTRACT
from realtime_framework.harnesses.default.prompts import STREAMING_FUNCTION_CALL_ACTION_SPACE
from realtime_framework.harnesses.default.prompts import STREAMING_FUNCTION_CALL_FRAME_PROMPT
from realtime_framework.harnesses.default.prompts import STREAMING_TEXT_ACTION_FRAME_PROMPT
from realtime_framework.harnesses.default.prompts import TEXT_ACTION_FRAME_PROMPT
from realtime_framework.harnesses.default.prompts import PromptOptions
from realtime_framework.harnesses.default.prompts import UNANSWERED_STEP_NOTE
from realtime_framework.harnesses.default.prompts import UNANSWERED_STEP_NOTES
from realtime_framework.harnesses.default.prompts import UNSUPPORTED_TERMINAL_ACTION_FEEDBACK
from realtime_framework.entity.actions import FrameGeometry
from realtime_framework.entity.actions import bind_action
from realtime_framework.sandbox.observation.som import FRAME_INTERACTABLES_HEADER


@dataclass(frozen=True)
class InteractionProfile:
    name: str
    action_space: str | None = None
    output_contract: str = STREAM_OUTPUT_CONTRACT
    # Its output contract under a finite B (PromptOptions.finite_budget), where that differs.
    finite_budget_output_contract: str | None = None
    frame_prompt: str = TEXT_ACTION_FRAME_PROMPT
    # A streaming profile's: how an action names its frame.
    frame_reference_rules: str | None = None

    def system_instruction(self, agent) -> str:
        """The episode's, composed for ``agent.prompt_options`` (harnesses/registry.py
        configure_agent), for which the profile was chosen (interaction_profile)."""
        output_contract = (
            self.finite_budget_output_contract
            if agent.prompt_options.finite_budget and self.finite_budget_output_contract
            else self.output_contract
        )
        if self.action_space is None:
            return gui_system_prompt(
                agent.task_instruction,
                scenario_prompt=getattr(agent, "system_prompt_override", None),
                clock_prompt=agent.streaming_note,
                frame_reference_rules=self.frame_reference_rules,
            )
        return "\n\n".join(
            filter(
                None,
                (
                    SCENARIO_ROLES[agent.task_category],
                    self.action_space,
                    self.frame_reference_rules,
                    output_contract,
                    getattr(agent, "streaming_note", None),
                    f"TASK:\n{agent.task_instruction}",
                ),
            )
        )

    def parse_action(self, agent, body: str) -> Action | None:
        return agent._parse_action(body)

    def parse_framed_action(self, agent, body: str) -> tuple[Action | None, object]:
        return agent._parse_framed_action(body)

    def bind_action(self, action: Action, geometry: FrameGeometry) -> Action:
        return bind_action(action, geometry, coordinate_space="model_image_px")

    def inspection_size(self, geometry: FrameGeometry) -> tuple[int, int]:
        return geometry.image_size


STANDARD_VLM_PROFILE = InteractionProfile("standard_vlm")
FUNCTION_CALL_PROFILE = InteractionProfile(
    "function_call",
    action_space=FUNCTION_CALL_ACTION_SPACE,
    output_contract=FUNCTION_CALL_OUTPUT_CONTRACT,
    finite_budget_output_contract=FINITE_BUDGET_FUNCTION_CALL_OUTPUT_CONTRACT,
    frame_prompt=FUNCTION_CALL_FRAME_PROMPT,
)
# The choice models' profiles.
JEV_PROFILE = InteractionProfile(
    "jev_decisions",
    action_space="Select one of the current observation's candidate actions.",
    output_contract=(
        "Return a typed Choice. You receive the current full interactables SoM and its "
        "action-bound mark legend, without screenshots. Choose only executable candidates. "
        "Typing choices contain observed literals, field suggestions, and calendar dates "
        "in or adjacent to observed months, filtered by explicit field constraints. "
        "Compose text lets you build other input through successive choices of exact words, "
        "characters, or Unicode code points, then finish to type it. Spaces and casing are explicit. "
        "You receive the current browser accessibility tree, visible control content, "
        "and JEV-specific appearance descriptors. Hidden application state is unavailable."
    ),
    frame_prompt="Choose the next action from the current observation.",
)

MULTIMODAL_CHOICE_PROFILE = InteractionProfile(
    "multimodal_decisions",
    action_space=JEV_PROFILE.action_space,
    output_contract=(
        "Return a typed Choice from the current observation's executable candidates. "
        "You receive its screenshot, full interactables SoM, action-bound mark legend, "
        "accessibility tree and visible control state. Hidden application state is unavailable. "
        "Typing choices contain observed literals and field suggestions. Compose text builds "
        "other input through successive choices of words, characters or Unicode code points, "
        "then finishes to type it. Spaces and casing are explicit."
    ),
    # Luna's transcript keeps the older frames, which JEV's digest leaves out.
    frame_prompt="Screen observed for this decision.",
)

MULTIMODAL_GAME_CHOICE_PROFILE = InteractionProfile(
    "multimodal_game_decisions",
    action_space=MULTIMODAL_CHOICE_PROFILE.action_space,
    output_contract=(
        MULTIMODAL_CHOICE_PROFILE.output_contract
        + " In games, declared keys offer taps and short holds, including simultaneous pairs. "
        "Holds persist for their stated duration; later actions do not cancel them. "
        "Grid choices name visible board positions without judging move legality. "
        "Pointer composition chooses coordinates before executing one action. "
        "mouse_down stays pressed until mouse_up; use move between them for a drag."
    ),
    frame_prompt=MULTIMODAL_CHOICE_PROFILE.frame_prompt,
)
# Context policy item 11 names the profile for each task category.
INTERACTION_PROFILES = {
    profile.name: profile
    for profile in (
        STANDARD_VLM_PROFILE,
        FUNCTION_CALL_PROFILE,
        JEV_PROFILE,
        MULTIMODAL_CHOICE_PROFILE,
        MULTIMODAL_GAME_CHOICE_PROFILE,
    )
}
# The profiles of a native-streaming run, whose drivers answer in text JSON or function calls.
# Every action names its frame and is read against it, however old, and its text JSON scenario
# prompt is streaming's as well (prompts.py system_prompt_for_category).
STREAMING_INTERACTION_PROFILES = {
    profile.name: profile
    for profile in (
        InteractionProfile(
            "standard_vlm",
            frame_prompt=STREAMING_TEXT_ACTION_FRAME_PROMPT,
            frame_reference_rules=FRAME_REFERENCE_RULES,
        ),
        InteractionProfile(
            "function_call",
            action_space=STREAMING_FUNCTION_CALL_ACTION_SPACE,
            output_contract=FUNCTION_CALL_OUTPUT_CONTRACT,
            frame_prompt=STREAMING_FUNCTION_CALL_FRAME_PROMPT,
            frame_reference_rules=FUNCTION_CALL_FRAME_REFERENCE_RULES,
        ),
    )
}


def interaction_profile(name: str, options: PromptOptions) -> InteractionProfile:
    """The profile context policy item 11 names, for a run of ``options``."""
    if options.interaction_mode == "streaming":
        return STREAMING_INTERACTION_PROFILES[name]
    return INTERACTION_PROFILES[name]


# Default harness
"""The benchmark-facing harness for every registered model API, composed with its driver."""

import uuid as uuid
from copy import deepcopy as deepcopy
from dataclasses import replace as replace

from realtime_framework.clock import SystemClock
from realtime_framework.harnesses.default.parsing import resolve_action_output
from realtime_framework.harnesses.default.parsing import resolve_structured_output
from realtime_framework.entity.model import GenerationOutput
from realtime_framework.entity.model import GenerationRequest
from realtime_framework.entity.model import PendingDecision
from realtime_framework.entity.events import ProviderEvent
from realtime_framework.entity.model import ProviderObservation
from realtime_framework.entity.model import RuntimeTurnResult
from realtime_framework.entity.model import SessionSeed
from realtime_framework.entity.model import SyncDisposition
from realtime_framework.entity.events import TextDelta
from realtime_framework.entity.events import UsageSnapshot
from realtime_framework.providers.registry import adapter_class
from realtime_framework.entity.events import EmptyInterruptedToolTurnError
from realtime_framework.entity.model import FATAL
from realtime_framework.entity.model import RequestRetry
from realtime_framework.entity.model import RetryCancelled
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.model import retry_cancellation
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.records import trace_scope


def _retry_usage_status(usages):
    if not usages:
        return "none"
    if all(u is not None for u in usages):
        return "complete"
    return "partial" if any(u is not None for u in usages) else "unavailable"
from realtime_framework.harnesses.default.parsing import FunctionCallTurnCollector
from realtime_framework.harnesses.default.context import RealtimeRuntime
from realtime_framework.harnesses.default.context import RequestExceedsContext
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.history import Accept
from realtime_framework.harnesses.default.history import Commit
from realtime_framework.harnesses.default.history import History
from realtime_framework.harnesses.default.history import Unanswered
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.request import AnswerRequirements
from realtime_framework.entity.request import CallParameters
from realtime_framework.entity.request import CanonicalRequest
from realtime_framework.harnesses.default.context import action_result_note
from realtime_framework.harnesses.default.context import native_output_limit
from realtime_framework.harnesses.default.frames import GROUNDING_FAILED
from realtime_framework.harnesses.default.frames import INVALID_GROUNDING_FRAME
from realtime_framework.harnesses.default.frames import FrameRecord
from realtime_framework.harnesses.default.frames import FrameTable
from realtime_framework.harnesses.default.request import build_request
from realtime_framework.harnesses.default.request import system_block
from realtime_framework.harnesses.jev.observation import choice_observation
from realtime_framework.sandbox.observation.capture import prepare_for_model
from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.sandbox.observation.som import observation_prompt_with_interactables
from realtime_framework.sandbox.observation.som import render_frame_number
from realtime_framework.sandbox.observation.som import resolve_action_references


class _GenerationOutputCollector:
    """Collect one provider response without retokenizing or slicing its text."""

    def __init__(self, agent):
        self.agent = agent
        self.visible: list[str] = []
        self.reasoning: list[str] = []
        self.usage: dict | None = None
        self.chunks: list[dict] = []
        self._native_output_tokens = 0
        self._native_counts_complete = True

    def consume(self, event: ProviderEvent) -> None:
        received_wall_time_ns = self.agent.clock.time_ns()
        received_monotonic_ns = self.agent.clock.monotonic_ns()
        if isinstance(event, UsageSnapshot):
            self.usage = event.usage
            return
        if not isinstance(event, TextDelta):
            return
        if event.native_token_ids:
            self._native_output_tokens += len(event.native_token_ids)
        else:
            self._native_counts_complete = False
        chunk = {
            "content": event.text,
            "channel": event.channel,
            "transport_event": event.raw_type,
            "sequence": event.sequence,
            "native_token_count": (len(event.native_token_ids) if event.native_token_ids else None),
            "received_wall_time_ns": received_wall_time_ns,
            "received_monotonic_ns": received_monotonic_ns,
            "receive_timestamp_source": "collector_consume",
        }
        self.chunks.append(chunk)
        if event.channel == "reasoning":
            self.reasoning.append(event.text)
            return
        self.visible.append(event.text)

    def finish(self) -> GenerationOutput:
        visible = "".join(self.visible)
        reasoning = "".join(self.reasoning)
        token_fields = self.agent._protocol_token_fields(self.usage)
        if self._native_counts_complete and self.visible:
            token_fields.update(
                {
                    "native_received_output_tokens": self._native_output_tokens,
                    "native_received_output_tokens_source": "native_streamed_token_ids",
                }
            )
        return GenerationOutput(
            text=visible,
            usage=self.usage,
            token_fields=token_fields,
            output_chunks=tuple(self.chunks),
            reasoning_text=reasoning,
        )


def bind_observation(agent, observation, prompt, tick):
    """Prepare one capture for the model, attach its read-only interactables and enter it in the
    frame table, whose record of it comes back too. Under streaming the frame's number is drawn
    on the screenshot, and the marks keep clear of it."""
    streaming = agent.interaction_mode == "streaming"
    prepared = prepare_for_model(observation.screenshot)
    metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
    expected = metadata.get("observation_id") or "missing-observation-id"
    width, height = agent.interaction_profile.inspection_size(prepared.geometry)
    execution = agent.tool_runtime.inspect(
        observation,
        requested_observation_id=expected,
        tick=getattr(observation, "tick_number", tick),
        model_width=width,
        model_height=height,
        model_image_bytes=prepared.data,
        model_image_mime_subtype=prepared.mime_subtype,
        # A streaming push enters its frame on the shell's one push lane, so no other frame
        # takes this number before the frame table gives it to this one.
        frame_number=agent.frames.next_frame() if streaming else None,
    )
    prompt = observation_prompt_with_interactables(
        prompt, execution.get("text") or "", FRAME_INTERACTABLES_HEADER
    )
    value = ProviderObservation(
        observation=observation,
        prompt=prompt,
        image_bytes=execution.get("image_bytes") or prepared.data,
        mime_type=f"image/{execution.get('image_mime_subtype') or prepared.mime_subtype}",
        image_detail=prepared.detail,
        geometry=prepared.geometry.to_dict(),
        inspection=execution["proof"],
    )
    frame = agent.frames.register(
        value.observation_id, observation.timestamp, value.geometry, execution["marks"]
    )
    if streaming:
        value.image_bytes = render_frame_number(value.image_bytes, frame.frame)
        value.mime_type = "image/png"
    return value, execution, frame


@dataclass(frozen=True)
class _Grounded:
    """One body of model output that parsed, read against its frame: the action as the model
    wrote it, and as the frame's marks resolve it, None when they cannot."""

    asked: Action
    action: Action | None
    frame: FrameRecord | None  # None: a streaming action whose frame cannot be found
    reference: object = None  # streaming: its frame field as written
    frame_error: str | None = None  # streaming: why its frame cannot be found
    resolution: dict | None = None  # how the marks resolved it; None when they were not read

    @property
    def unknown_mark(self) -> str | None:
        """The mark its frame does not have, which stopped its resolution: resolution stops at
        the first reference that fails."""
        if self.action is not None or self.resolution is None:
            return None
        failed = (self.resolution.get("resolutions") or [{}])[-1]
        return failed["mark"] if failed.get("error") == "mark_not_in_observation" else None


def parse_grounded_action(agent, text, arrived_unix_s, own_frame):
    """Parse one action from model output and interpret it against its frame-table entry: the
    frame's marks resolve its element references and its geometry binds its coordinates. The
    action, its protocol, and the last resolution the output made, which the inspection's tool
    trace records.

    A streaming action names its frame, however old, and is rejected (no action) when that frame
    cannot be found; the frame and its age are recorded. A turn-based action uses ``own_frame``,
    the one its generation sent: an abandoned generation may enter its frame later.

    Output whose one action names a mark its frame does not have is a grounding failure, not
    output without an action: the protocol records the action asked for and the mark.

    A function call or a Decision's choice is the action object itself, which is read as it is;
    only text JSON is recovered from the forms models wrap it in.
    """
    frames = agent.frames
    streaming = agent.interaction_mode == "streaming"
    structured = agent.driver.capabilities.action_channel != "text_json"

    def ground(body) -> _Grounded | None:
        if structured:
            asked, reference = structured_action(body)
        elif streaming:
            asked, reference = agent.interaction_profile.parse_framed_action(agent, body)
        else:
            asked, reference = agent.interaction_profile.parse_action(agent, body), None
        frame, error = frames.locate(reference) if streaming else (own_frame, None)
        if asked is None:
            return None
        if frame is None:
            return _Grounded(asked, asked, frame, reference, error)
        action, resolution = resolve_action_references(asked, frame.marks)
        if streaming:
            # The inspection row names the frame whose marks resolved it.
            resolution = {
                **resolution,
                "frame": frame.frame,
                "observation_id": frame.observation_id,
            }
        return _Grounded(asked, action, frame, reference, error, resolution)

    output = (
        resolve_structured_output(text, ground)
        if structured
        else resolve_action_output(text, ground, framed=streaming)
    )
    decision = output.decision
    resolutions = [parsed.resolution for parsed in output.parses if parsed.resolution is not None]
    grounding = last = resolutions[-1] if resolutions else None
    action_protocol = decision.metadata()
    unknown = next((parsed for parsed in output.parses if parsed.unknown_mark is not None), None)
    if decision.protocol_failure == "invalid_action_output" and unknown is not None:
        grounding = unknown.resolution
        action_protocol.update(
            protocol_failure=GROUNDING_FAILED,
            requested=action_text(unknown.asked),
            unknown_mark=unknown.unknown_mark,
        )
    frame = own_frame
    if streaming and decision.valid:
        chosen = output.chosen
        frame = chosen.frame
        if frame is None:
            action_protocol.update(
                rejection=INVALID_GROUNDING_FRAME,
                frame_reference={"frame": chosen.reference, "error": chosen.frame_error},
            )
            return None, action_protocol, last
        action_protocol["frame_reference"] = {
            "frame": frame.frame,
            "observation_id": frame.observation_id,
            **frames.age(frame, arrived_unix_s),
        }
    action = agent.interaction_profile.bind_action(
        decision.action, FrameGeometry.from_dict(frame.geometry)
    )
    action_protocol["grounding"] = grounding
    return action, action_protocol, last


def structured_action(body: str) -> tuple[Action | None, object]:
    """The action of a function call or a Decision's choice, from the JSON action object its
    channel gave (FunctionCallTurnCollector, the streaming agent's calls, the Decision encoder's
    choice), and the value of its frame field."""
    data = json.loads(body)
    reference = data.pop("frame", None)
    return ActionTranslator.from_json(data), reference


def action_text(action: Action) -> str:
    """An action as the action record and the feedback name it."""
    return action.action_type.value + (f" {action.parameters}" if action.parameters else "")


@dataclass
class _Staged:
    """What accepting a generation's result writes into the history. Discarding it writes no
    history: it ends the tool calls and records the usage of a result that came back, or of the
    questions a Decision cancelled before its result answered."""

    unattached: str  # the feedback no turn carries when it started
    inspections: list  # its observation's tool calls, still open
    result: RuntimeTurnResult | None = None  # None when it was cancelled before any result
    observation: ProviderObservation | None = None  # the frame its action is based on
    record: dict | None = None  # its action record, numbered when written
    # A cancelled generation's id, and the usage of the Decision questions it answered.
    generation_id: str | None = None
    partial_usage: dict | None = None


@dataclass
class _Generation:
    """One kernel generation, from whichever of StartGeneration, AbortGeneration and
    DiscardResult reaches the agent first until it is accepted, or discarded and dropped. An
    abandoned generation can still be running when the next one starts, so it works on this
    alone; what the agent keeps of a generation is set when it starts (DefaultHarness._start)."""

    number: int  # the kernel's generation number
    cancel: threading.Event  # set by abort() and close(); the runtime stops the generation
    # "waiting" until its worker starts generate(), "running" until generate() returns, then
    # "returned"; "discarded" once discard() came before it returned, to drop it then.
    state: str = "waiting"
    decision: int | None = None
    # The decision's output caps (DecisionBudget).
    provider_output_tokens: int | None = None
    transport_output_tokens: int | str | None = None
    # (ChoiceRequirements, SelectionMaterial) of its observation, for its requests.
    choice: tuple | None = None
    frame: FrameRecord | None = None  # its observation's frame-table record
    staged: _Staged | None = None


class DefaultHarness(VisionAgentBase):
    """Common frame, context, action, and audit behavior for every catalog model.

    The registered driver for ModelConfig.adapter_type supplies the protocol; its capabilities
    select history transport, the action channel, and interruption behavior. The context policy
    selects the interaction profile (item 11) and the observation form (item 12).
    """

    interaction_profile: InteractionProfile  # set by initialize() from context policy item 11
    supports_unbounded_output = True
    # A streaming run builds NativeStreamingAgent instead (harnesses/registry.py create_agent).
    interaction_mode = "turn_based"

    def __init__(self, config, log_dir=None, *, trace, clock=None):
        self._provider_usage_started = False
        # The generations the kernel named that are not settled, by number; abort() and close()
        # cancel them.
        self._generations: dict[int, _Generation] = {}
        self._generations_lock = threading.Lock()
        # The episode's operational clock; the runtime, the driver and the retry wait read it.
        self.clock = clock or SystemClock()
        super().__init__(config, log_dir, trace=trace)

    @property
    def driver(self):
        runtime = getattr(self, "_runtime", None)
        return runtime.driver if runtime is not None else self._driver

    @driver.setter
    def driver(self, value):
        self._driver = value

    def _setup_backend(self):
        self.driver = adapter_class(self.config.adapter_type)(self.config, clock=self.clock)
        self.driver.bind_trace(self._trace)
        self._context_policy = current_context_policy().for_path(
            self.config.adapter_type, self.config.provider_options, self.config.context_preset
        )
        # Started by initialize(); close() stops it once the runtime has closed the session.
        self._execution_thread = ExecutionThread(
            self.clock, f"execution-thread-{self.driver.capabilities.protocol_family}"
        )
        self.history = History(self._context_policy)
        self._runtime = RealtimeRuntime(
            self.driver,
            self._context_policy,
            history=self.history,
            trace=self._trace,
            execution=self._execution_thread,
            clock=self.clock,
        )

    def _record_provider_usage(self, staged: _Staged, tick, *, stale):
        """Audit harness-visible generations, including providers with inline usage; ``stale``
        when the result was discarded. A Decision cancelled before its result has only the usage
        of the questions it answered."""
        result = staged.result
        if result is None:
            generation_id, usage, missing_reason = staged.generation_id, staged.partial_usage, None
        else:
            generation_id, usage = result.generation_id, result.output.usage
            missing_reason = result.output.token_fields.get("provider_usage_missing_reason")
        usage = deepcopy(usage)
        record = {
            "schema": "provider-usage-v1",
            "generation_id": generation_id,
            "observation_id": staged.observation.observation_id,
            "tick": tick,
            "provider_protocol_family": self.driver.capabilities.protocol_family,
            "provider_upstream": getattr(self.config, "upstream_provider", None),
            "collection": "inline",
            "status": "reported" if usage else "missing",
            "stale": stale,
            "usage": usage,
            "usage_at_action_commit": deepcopy(usage),
            "missing_reason": (
                None if usage else missing_reason or "backend_did_not_return_usage"
            ),
            "recorded_unix_s": self.clock.time(),
            "scope": "harness_generation",
        }
        if self.log_dir:
            mode = "a" if self._provider_usage_started else "w"
            self._provider_usage_started = True
            with self._log_lock, (self.log_dir / "provider-usage.jsonl").open(mode) as stream:
                stream.write(json.dumps(record) + "\n")

    def initialize(self, task_instruction, initial_observation):
        super().initialize(task_instruction, initial_observation)
        self.frames = FrameTable()
        self.interaction_profile = interaction_profile(
            self._context_policy.prompts[self.task_category],  # context policy item 11
            self.prompt_options,
        )
        fixed = self.interaction_profile.system_instruction(self)
        # Its inputs are set before initialize() and never change during the episode.
        self._system_block = system_block(fixed, self.clock.time_ns())
        self._answer = self._answer_requirements()
        self._execution_thread.start()
        with trace_scope(phase="initialize"):
            self._runtime.start(
                SessionSeed(
                    system_instruction=fixed,
                    model_name=self.model_name,
                    native_output_limit=self.config.native_output_limit,
                    fps=float(self.config.fps),
                    answer=self._answer,
                )
            )

    def discard_pending_observations(self):
        """No frame waits in the agent: generate() is handed its observation."""

    def transport_output_token_cap(self, provider_cap):
        return self.driver.transport_output_token_cap(provider_cap)

    def _prepare_observations(self, observations, tick):
        """The prepared observations, their inspections, the newest one's answer options, and
        the newest one's frame-table record: the frame the action is based on."""
        prepared_values = []
        inspections = []
        try:
            for observation in observations:
                prompt = self.interaction_profile.frame_prompt
                prepared, execution, frame = bind_observation(self, observation, prompt, tick)
                prepared_values.append(prepared)
                inspections.append(execution)
        except Exception:
            for execution in inspections:
                try:
                    self.tool_runtime.finish(execution, outcome="observation_preparation_error")
                except Exception:  # noqa: BLE001, S110 - preserve the preparation failure
                    pass
            raise
        # (ChoiceRequirements, SelectionMaterial) of this decision's observation, for the request.
        choice = None
        if self._context_policy.observation != "som_screenshot":  # context policy item 12
            for value, execution in zip(prepared_values, inspections):
                choice = choice_observation(
                    self.task_instruction,
                    value,
                    execution,
                    frame_prompt=self.interaction_profile.frame_prompt,
                    game=self.task_category == "browser_game",
                )
        return prepared_values, inspections, choice, frame

    def _answer_requirements(self) -> AnswerRequirements:
        channel = self.driver.capabilities.action_channel
        return AnswerRequirements(
            action_channel=channel,
            format_profile=self.interaction_profile.name,
            function_declarations=(
                tuple(gui_function_declarations()) if channel == "function_call" else ()
            ),
        )

    def _build_request(self, turns, ledger, call, choice) -> CanonicalRequest:
        answer, selection = self._answer, None
        if choice is not None:
            # The one observation this generation prepared (the runtime takes exactly one per
            # generation); its candidates are the answer options.
            requirements, selection = choice
            answer = replace(answer, choice=requirements)
        return build_request(
            self._context_policy, self._system_block, turns, ledger, answer, call, selection
        )

    @staticmethod
    def _cancelled_result(latency: float, usage: dict | None = None) -> dict:
        """The result of a generation an abort cancelled: of a Decision given up on after some
        of its questions were answered, their usage, without the question asked when it was."""
        result = {
            "action": None,
            "thinking": "",
            "raw": "",
            "latency": latency,
            "usage": usage,
            "stale": True,
        }
        if usage is not None:
            result["provider_usage_missing_reason"] = "cancelled_request_usage_not_reported"
        return result

    def _invalid_observation_result(self, proof, latency=0.0):
        protocol = {
            "source": "invalid_action_fallback_wait",
            "valid_action": False,
            "action_text": None,
            "protocol_failure": "inspection_protocol_error",
        }
        return {
            "action": Action.wait(),
            "thinking": "",
            "raw": "",
            "latency": latency,
            "usage": None,
            "inspection": proof,
            "action_protocol": protocol,
            "invalid_category": "harness",
            "invalid_reason": "inspection_protocol_error",
        }

    def _generation(self, number: int) -> _Generation:
        """Generation ``number``'s record, made by whichever call names it first. The caller
        holds the lock."""
        if number not in self._generations:
            self._generations[number] = _Generation(number, self.clock.event())
        return self._generations[number]

    def _start(self, number: int, decision: int, budget: DecisionBudget) -> _Generation:
        """Start generation ``number``, unless it was discarded first. The agent keeps the newest
        generation's action cap, which the runner audits against the decision's allowance once
        the result is back. A generation aborted before its worker got here does not keep it: it
        may start after the next one did."""
        with self._generations_lock:
            generation = self._generation(number)
            generation.decision = decision
            generation.provider_output_tokens = budget.provider_output_tokens
            generation.transport_output_tokens = budget.transport_output_tokens
            if not generation.cancel.is_set():
                self._frame_action_max_tokens = native_output_limit(
                    self.config.native_output_limit, budget.provider_output_tokens
                )
            if generation.state == "waiting":
                generation.state = "running"
        return generation

    def _cancel(self, number: int | None = None) -> list[_Generation]:
        """Cancel the kernel's generation ``number``, or every generation, and return those of
        them that started. The kernel can abort a generation before its worker has started
        generate(); that generation then starts cancelled. One discarded before it returned is
        left to end."""
        with self._generations_lock:
            named = (
                list(self._generations.values()) if number is None else [self._generation(number)]
            )
            cancelled = [generation for generation in named if generation.state != "discarded"]
            for generation in cancelled:
                generation.cancel.set()
        return [generation for generation in cancelled if generation.state != "waiting"]

    def generate(
        self, generation: int, decision: int, observation, budget: DecisionBudget
    ) -> PendingDecision:
        """StartGeneration, on a worker thread. The result is held, not written: accept() writes
        it into the history and discard() drops it."""
        state = self._start(generation, decision, budget)
        try:
            result = self._generate(state, [observation])
        finally:
            with self._generations_lock:
                discarded = state.state == "discarded"
                if discarded:
                    del self._generations[generation]
                else:
                    state.state = "returned"
            if discarded:
                self._drop(state)
        return PendingDecision(generation, decision, result)

    def accept(self, generation: int) -> None:
        with self._generations_lock:
            state = self._generations.pop(generation)
        self._write(state)

    def discard(self, generation: int) -> None:
        """DiscardResult. At the end of an episode the shell also discards a generation whose
        worker is still running, or has not started generate() yet, which the kernel has
        aborted by then: it is dropped as generate() returns."""
        with self._generations_lock:
            state = self._generation(generation)
            if state.state != "returned":
                state.state = "discarded"
                return
            del self._generations[generation]
        self._drop(state)

    def _write(self, generation: _Generation) -> None:
        """An accepted result joins the history: its turn, its action record and its usage."""
        staged = generation.staged
        if staged is None:
            return
        result = staged.result
        self._record_provider_usage(staged, generation.decision, stale=False)
        self._runtime.accept(
            Accept(generation.decision, result.turn, staged.record, staged.unattached)
        )
        self._accumulate_usage(result.output.usage)
        self._end_tool_calls(
            staged, "action_protocol_error" if staged.record["protocol_failure"] else "completed"
        )

    def _drop(self, generation: _Generation) -> None:
        """A stale result writes nothing into the history."""
        staged = generation.staged
        if staged is None:
            return
        if staged.result is not None or staged.partial_usage is not None:
            self._record_provider_usage(staged, generation.decision, stale=True)
        self._end_tool_calls(staged, "superseded_by_timeout")

    def _end_tool_calls(self, staged: _Staged, outcome: str) -> None:
        result = staged.result
        latency, text = (0.0, "") if result is None else (result.latency_s, result.output.text)
        for index, execution in enumerate(staged.inspections):
            self.tool_runtime.finish(
                execution,
                action_latency_s=latency,
                raw_action_output=text if index == len(staged.inspections) - 1 else "",
                outcome=outcome,
            )

    def _generate(self, generation: _Generation, observations: list) -> dict:
        """One generation: its decision result, marked stale when an abort cancelled it. Nothing
        is written into the history; what accepting the result writes is kept in
        ``generation.staged``."""
        tick = generation.decision
        if generation.cancel.is_set():
            # Aborted before its worker got here: nothing is inspected, entered or sent.
            return self._cancelled_result(0.0)
        started = self.clock.monotonic()
        unattached = self.history.unattached
        try:
            prepared, inspection_executions, generation.choice, generation.frame = (
                self._prepare_observations(observations, tick)
            )
        except Exception as exc:  # noqa: BLE001 - preparation faults become invalid-run metadata
            latency = self.clock.monotonic() - started
            detail = repr(exc)[:500]
            log_entry = {
                "type": "decision",
                "tick": tick,
                "decision": tick,
                "latency_s": round(latency, 2),
                "decision_wall_latency_s": latency,
                "thinking": "",
                "model_output": "",
                "parsed_action": None,
                "error": detail,
            }
            return {
                "action": None,
                "thinking": "",
                "raw": "",
                "latency": latency,
                "decision_wall_latency_s": latency,
                "usage": None,
                "invalid_category": "harness",
                "invalid_reason": "observation_preparation_error",
                "invalid_detail": detail,
                "_trajectory_entry": log_entry,
            }
        newest_proof = prepared[-1].inspection
        invalid_proof = next(
            (value.inspection for value in prepared if not value.inspection.get("valid")), None
        )
        if invalid_proof is not None:
            for execution in inspection_executions:
                self.tool_runtime.finish(execution, outcome="inspection_protocol_error")
            latency = self.clock.monotonic() - started
            result = self._invalid_observation_result(invalid_proof, latency)
            result["decision_wall_latency_s"] = latency
            result["_trajectory_entry"] = {
                "type": "decision",
                "tick": tick,
                "decision": tick,
                "latency_s": round(latency, 2),
                "decision_wall_latency_s": latency,
                "thinking": "",
                "model_output": "",
                "parsed_action": {"action_type": "wait", "parameters": {}},
                "inspection": invalid_proof,
            }
            return result

        configured_provider_cap = generation.provider_output_tokens
        provider_cap = None if configured_provider_cap is None else int(configured_provider_cap)
        configured_transport_cap = generation.transport_output_tokens
        transport_cap = (
            self.transport_output_token_cap(provider_cap)
            if configured_transport_cap is None
            else configured_transport_cap
        )
        request = GenerationRequest(
            generation_id=f"turn-{tick}-{uuid.uuid4().hex[:12]}",
            transport_output_token_cap=transport_cap,
            generation=generation.number,
        )
        call = CallParameters(
            model=self.config.model_name,
            reasoning_effort=self.config.reasoning_effort,
            output_token_cap=provider_cap,
        )

        def run_once(active_collector):
            return self._runtime.run_turn(
                prepared,
                request,
                active_collector,
                unattached=unattached,
                timeout=None,
                request_builder=lambda turns, ledger: self._build_request(
                    turns, ledger, call, generation.choice
                ),
                decision=tick,
                cancel=generation.cancel,
            )

        collector_type = (
            FunctionCallTurnCollector
            if self.driver.capabilities.action_channel == "function_call"
            else _GenerationOutputCollector
        )
        classifier = self.driver.classify_transport_error
        collector = None
        retry_reasons = []
        retry_delays = []
        retry_usages = []
        failure_detail = ""
        failure_reason = ""
        final_error = None
        failure_usage = partial_usage = None

        def send(attempt):
            nonlocal collector
            collector = collector_type(self)
            with self._attempt_scope(generation, request, attempt):
                return run_once(collector)

        def classify(exc):
            if isinstance(exc, RequestExceedsContext):  # the harness's own; a retry cannot fit
                return TransportFailure(FATAL)
            try:
                return classifier(exc)
            except Exception:  # noqa: BLE001, S110 - unclassifiable stays fatal
                pass
            return TransportFailure(FATAL)

        def received_output(exc):
            return not (
                not collector.visible
                and not collector.reasoning
                and (collector.usage is None or isinstance(exc, EmptyInterruptedToolTurnError))
                and not getattr(collector, "calls", ())
                and not getattr(collector, "received_native_output", False)
            )

        def record(attempt, exc, failure, delay):
            nonlocal failure_detail, failure_reason, final_error, failure_usage, partial_usage
            received = received_output(exc)
            if delay is not None:
                retry_reasons.append(f"{failure.kind}: {exc!r}"[:300])
                retry_usages.append(collector.usage)
                retry_delays.append(delay)
                self._attempt_failed(
                    generation,
                    request,
                    attempt,
                    exc,
                    failure,
                    received,
                    collector.usage,
                    delay=delay,
                )
                return
            # A partially received turn cannot be replayed without risking a duplicate action,
            # whatever the cause.
            final_error = exc
            failure_reason = "provider_stream_interrupted" if received else failure.invalid_reason
            if isinstance(exc, RequestExceedsContext):  # the generation request was not sent
                failure_reason = exc.invalid_reason
            failure_detail = repr(exc)[:500]
            if retry_reasons:
                failure_detail = (
                    f"{len(retry_reasons)} transport retries exhausted: "
                    f"{'; '.join(retry_reasons)}; final error: {exc!r}"
                )[:500]
            # A Decision that failed after some of its questions were answered carries their
            # usage (providers/decisions/driver.py); the failed question's is not known.
            partial_usage = getattr(exc, "partial_usage", None)
            failure_usage = collector.usage if partial_usage is None else partial_usage
            self._attempt_failed(
                generation,
                request,
                attempt,
                exc,
                failure,
                received,
                failure_usage,
                invalid_reason=failure_reason,
            )

        retry = RequestRetry(
            f"{self._trace.attempt_id}:{prepared[-1].observation_id}:turn",
            classify,
            record,
            cancelled=lambda attempt, exc: self._attempt_cancelled(
                generation, request, attempt, exc
            ),
            replayable=lambda exc: not received_output(exc),
            clock=self.clock,
        )
        try:
            with retry_cancellation(generation.cancel):
                runtime_result = retry.run(send)
        except RetryCancelled as exc:
            # A Decision given up on after some of its questions were answered carries their
            # usage, which discarding it records as stale, and the shell as what it spent
            # (decision 37).
            answered = getattr(exc, "partial_usage", None)
            generation.staged = _Staged(
                unattached,
                inspection_executions,
                observation=prepared[-1],
                generation_id=request.generation_id,
                partial_usage=answered,
            )
            return self._cancelled_result(self.clock.monotonic() - started, answered)
        except Exception as exc:
            # Only the failure record() kept as final becomes an invalid decision; anything else
            # raised on the way (by record() itself, the backoff, the output check) propagates,
            # and the shell ends the run with it as generation_error, the final failure's record
            # included.
            if exc is not final_error:
                raise
        else:
            self._attempt_finished(generation, request, len(retry_reasons) + 1, runtime_result)
            if retry_reasons:
                runtime_result.provider_metadata.update(
                    {
                        "transport_reconnect_attempts": len(retry_reasons),
                        "transport_reconnect_reason": retry_reasons[0],
                        "transport_reconnect_reasons": retry_reasons,
                        "transport_reconnect_backoff_s": retry_delays,
                        "transport_reconnect_replayed_from_canonical_history": True,
                        "transport_retry_usage": retry_usages,
                        "transport_retry_usage_status": _retry_usage_status(retry_usages),
                    }
                )

        if failure_detail:
            latency = self.clock.monotonic() - started
            for execution in inspection_executions:
                self.tool_runtime.finish(
                    execution, action_latency_s=latency, outcome="action_round_error"
                )
            log_entry = {
                "type": "decision",
                "tick": tick,
                "decision": tick,
                "latency_s": round(latency, 2),
                "decision_wall_latency_s": latency,
                "thinking": "",
                "model_output": "",
                "parsed_action": None,
                "error": failure_detail,
                "usage": failure_usage,
                "output_chunks": collector.chunks,
                "transport_reconnect_attempted": bool(retry_reasons),
                "transport_reconnect_attempts": len(retry_reasons),
                "transport_reconnect_backoff_s": retry_delays,
                "transport_failure_reason": failure_reason,
                "transport_retry_usage": retry_usages,
                "transport_retry_usage_status": _retry_usage_status(retry_usages),
            }
            result = {
                "action": None,
                "thinking": "",
                "raw": "",
                "latency": latency,
                "decision_wall_latency_s": latency,
                "usage": failure_usage,
                "output_chunks": collector.chunks,
                "invalid_category": "provider",
                "invalid_reason": failure_reason,
                "invalid_detail": failure_detail,
                "_trajectory_entry": log_entry,
            }
            if isinstance(final_error, RequestExceedsContext):
                # Its token record (evaluation/clock.py token_record) says why it has no usage,
                # and its clock output count stays unknown, with no clock source, rather than the
                # zero of the frame's usage delta (evaluation/episode_records.py
                # clock_output_tokens). It still counts as one inference response.
                result["provider_usage_missing_reason"] = "request_not_sent"
            if partial_usage is not None:
                # Its token record is partial: the usage is known, not complete (decision 19).
                result["provider_usage_missing_reason"] = "failed_request_usage_not_reported"
            return result

        output = runtime_result.output
        action_observation = prepared[-1]
        newest_execution = inspection_executions[-1]
        newest_proof = action_observation.inspection

        native_protocol_failure = output.token_fields.get("protocol_failure")
        if native_protocol_failure:
            action = Action.wait()
            action_protocol = {
                "source": "invalid_action_fallback_wait",
                "valid_action": False,
                "action_text": None,
                "protocol_failure": native_protocol_failure,
                "parse_mode": (
                    None
                    if output.token_fields.get("function_validation_error")
                    else "native_function_call_count"
                ),
                "candidate_count": output.token_fields.get("function_call_count", 0),
            }
            action = self.interaction_profile.bind_action(
                action, FrameGeometry.from_dict(action_observation.geometry)
            )
            action_protocol["grounding"] = newest_execution.get("reference_resolution")
        else:
            action, action_protocol, resolution = parse_grounded_action(
                self, output.text, self.clock.time(), generation.frame
            )
            if resolution is not None:
                newest_execution["reference_resolution"] = resolution
        thinking = output.reasoning_text
        serialization = output.token_fields.get("action_serialization")
        if serialization in {"native_function_call", "native_function_calls"}:
            action_protocol.update(
                {
                    "serialization": serialization,
                    "submission_boundary": output.token_fields.get(
                        "submission_boundary", "native_turn_complete"
                    ),
                    "function_call_id": output.token_fields.get("function_call_id"),
                    "function_call_count": output.token_fields.get("function_call_count"),
                    "function_call_ids": output.token_fields.get("function_call_ids"),
                    "function_call_names": output.token_fields.get("function_call_names"),
                }
            )
            if not native_protocol_failure:
                action_protocol["source"] = "native_function_call"
        protocol_failure = action_protocol.get("protocol_failure")
        record = {
            "action": output.text,
            "thinking": thinking,
            # A grounding failure asked for the action it named; other output that is no action
            # asked for no valid one.
            "requested": action_protocol.get("requested")
            or ("invalid_action" if protocol_failure else action_text(action)),
            "observation_tick": action_observation.observation.tick_number,
            "observation_timestamp": action_observation.observation.timestamp,
            "parse_failed": bool(protocol_failure),
            "protocol_failure": protocol_failure,
            "parse_mode": action_protocol.get("parse_mode"),
        }
        if protocol_failure == GROUNDING_FAILED:
            record["unknown_mark"] = action_protocol["unknown_mark"]
        generation.staged = _Staged(
            unattached,
            inspection_executions,
            runtime_result,
            action_observation,
            record,
        )

        token_fields = dict(output.token_fields)
        token_fields["backend_route"] = runtime_result.provider_metadata.get("backend_route")
        token_fields["provider_protocol_family"] = self.driver.capabilities.protocol_family
        token_fields["provider_upstream"] = getattr(self.config, "upstream_provider", None)

        sync = runtime_result.history_sync
        runtime_metadata = {
            "runtime_state": self._runtime.state.value,
            "protocol_family": self.driver.capabilities.protocol_family,
            "generation_id": runtime_result.generation_id,
            "interrupted": runtime_result.interrupted,
            "terminal_history_state": runtime_result.terminal_history_state.value,
            "history_sync": {
                "disposition": sync.disposition.value,
                "history_state": sync.history_state.value,
                "detail": sync.detail,
                **sync.provider_metadata,
            },
            **runtime_result.provider_metadata,
        }
        decision_wall_latency_s = self.clock.monotonic() - started
        request_started_ns = runtime_result.provider_metadata.get(
            "provider_call_started_monotonic_ns"
        )
        output_chunks = list(output.output_chunks)
        first_output_received_ns = next(
            (
                chunk.get("received_monotonic_ns")
                for chunk in output_chunks
                if isinstance(chunk, dict)
                and isinstance(chunk.get("received_monotonic_ns"), int)
                and (chunk.get("content") not in {None, ""} or chunk.get("function_call"))
            ),
            None,
        )
        provider_ttft_s = (
            (first_output_received_ns - request_started_ns) / 1_000_000_000
            if isinstance(request_started_ns, int)
            and isinstance(first_output_received_ns, int)
            and first_output_received_ns >= request_started_ns
            else None
        )
        provider_ttft_source = (
            "provider_call_start_to_first_output_event_collector_receive_monotonic"
            if provider_ttft_s is not None
            else None
        )
        log_entry = {
            "type": "decision_discarded" if runtime_result.cancelled else "decision",
            "tick": tick,
            "decision": tick,
            "num_frames": len(prepared),
            "latency_s": round(decision_wall_latency_s, 2),
            "decision_wall_latency_s": decision_wall_latency_s,
            "provider_call_wall_latency_s": runtime_result.provider_call_latency_s,
            "runtime_turn_wall_latency_s": runtime_result.latency_s,
            "provider_ttft_s": provider_ttft_s,
            "provider_ttft_source": provider_ttft_source,
            "provider_ttft_missing_reason": (
                None if provider_ttft_s is not None else "no_timestamped_output_event"
            ),
            "thinking": thinking,
            "model_output": output.text,
            "parsed_action": {
                "action_type": action.action_type.value,
                "parameters": action.parameters,
            },
            "action_protocol": action_protocol,
            "inspection": newest_proof,
            "usage": output.usage,
            "realtime_runtime": runtime_metadata,
            **token_fields,
        }
        result = {
            "action": None if runtime_result.cancelled else action,
            "thinking": thinking,
            "raw": output.text,
            # Consumers read `latency` as the complete decision wall duration.
            "latency": decision_wall_latency_s,
            "decision_wall_latency_s": decision_wall_latency_s,
            "provider_call_wall_latency_s": runtime_result.provider_call_latency_s,
            "runtime_turn_wall_latency_s": runtime_result.latency_s,
            "provider_ttft_s": provider_ttft_s,
            "provider_ttft_source": provider_ttft_source,
            "provider_ttft_missing_reason": (
                None if provider_ttft_s is not None else "no_timestamped_output_event"
            ),
            "usage": output.usage,
            "inspection": newest_proof,
            "action_protocol": action_protocol,
            "output_chunks": output_chunks,
            "realtime_runtime": runtime_metadata,
            "provider_response_status": runtime_metadata.get("provider_response_status"),
            "provider_response_status_reason": runtime_metadata.get(
                "provider_response_status_reason"
            ),
            "_trajectory_entry": log_entry,
            **token_fields,
        }
        if runtime_result.cancelled:
            result["stale"] = True
        if sync.disposition in {SyncDisposition.UNSUPPORTED, SyncDisposition.BROKEN}:
            result.update(
                {
                    "invalid_category": "provider",
                    "invalid_reason": "history_sync_failed",
                    "invalid_detail": sync.detail,
                }
            )
        return result

    def commit(self, outcome):
        """Commit: report decision ``outcome.decision`` on its own action record and turn. A
        decision whose result was never accepted, a move over the per-move cap, has neither: no
        earlier record changes, and its feedback, which names no action, opens the next frame
        whatever the policy's feedback placement."""
        record = self.history.action(outcome.decision)
        if record is None:
            feedback = f"EXECUTION RESULT: status={outcome.status}."
            if outcome.detail:
                feedback += f" Detail: {outcome.detail}"
            note = UNANSWERED_STEP_NOTES.get(outcome.status, UNANSWERED_STEP_NOTE)
            self.history.write(Unanswered(feedback + " " + note))
            return
        # The loop has no terminal actions: the kernel ran a done or fail as this wait.
        unsupported = outcome.protocol_failure == "unsupported_terminal_action"
        result, feedback = self._report(
            record, outcome.status, outcome.detail, outcome.action if unsupported else None
        )
        self.history.write(Commit(outcome.decision, result, feedback, self.clock.time_ns()))

    @staticmethod
    def _report(record: dict, status: str, detail: str, replacement=None) -> tuple[dict, str]:
        """An action record's outcome: what ran and how, as the record takes them, and the
        outcome's feedback. ``replacement`` is the wait the kernel ran in place of a done or
        fail."""
        parse_failed = record["parse_failed"]
        requested = record["requested"]
        landed_statuses = {"applied", "wait", "terminal", "terminal_ignored"}
        executed = "wait()" if parse_failed or status not in landed_statuses else requested
        replaced = replacement is not None and status in landed_statuses
        if replaced:
            executed = action_text(replacement)
        result = {"applied": status == "applied", "result": status, "executed": executed}
        feedback = f"EXECUTION RESULT: actual={executed}; requested={requested}; status={status}."
        if detail:
            feedback += f" Detail: {detail}"
        if "unknown_mark" in record:
            note = (
                GROUNDING_FAILED_FEEDBACK.format(mark=record["unknown_mark"])
                + f" ({GROUNDING_FAILED})"
            )
        else:
            note = action_result_note(
                "none" if parse_failed else status,
                record["protocol_failure"] if parse_failed else detail,
            )
        if note:
            feedback += " " + note
        if replaced:
            feedback += " " + UNSUPPORTED_TERMINAL_ACTION_FEEDBACK
        return result, feedback

    def _attempt_scope(self, generation, request, attempt):
        return trace_scope(
            phase="decision",
            decision=generation.decision,
            generation=generation.number,
            generation_id=request.generation_id,
            attempt=attempt,
        )

    def _attempt_finished(self, generation, request, attempt, result) -> None:
        """Record one completed attempt of the shared whole-turn retry."""
        metadata = result.provider_metadata
        with self._attempt_scope(generation, request, attempt):
            self._trace.capture(
                TraceEvent.MODEL_ATTEMPT_FINISHED,
                {
                    "kind": "generation",
                    "outcome": "cancelled" if result.cancelled else "ok",
                    "usage": result.output.usage,
                    "token_fields": result.output.token_fields,
                    "runtime_turn_wall_latency_s": result.latency_s,
                    "provider_call_wall_latency_s": result.provider_call_latency_s,
                    "provider_call_started_monotonic_ns": metadata.get(
                        "provider_call_started_monotonic_ns"
                    ),
                    "terminal_history_state": result.terminal_history_state.value,
                    "interrupted": result.interrupted,
                    "cancelled": result.cancelled,
                    "backend_route": metadata.get("backend_route"),
                },
            )

    def _attempt_failed(
        self,
        generation,
        request,
        attempt,
        exc,
        failure,
        received_output,
        usage,
        *,
        delay=None,
        invalid_reason=None,
    ) -> None:
        """Record one failed attempt: retried after ``delay``, or final with ``invalid_reason``."""
        with self._attempt_scope(generation, request, attempt):
            self._trace.capture(
                TraceEvent.MODEL_ATTEMPT_FINISHED,
                {
                    "kind": "generation",
                    "outcome": "retry" if delay is not None else "failed",
                    "failure_kind": failure.kind,
                    "invalid_reason": invalid_reason,
                    "error": repr(exc)[:500],
                    "received_output": received_output,
                    "retry_delay_s": delay,
                    "usage": usage,
                },
            )

    def _attempt_cancelled(self, generation, request, attempt, exc) -> None:
        """Record the attempt the generation's cancellation stopped: one not yet sent, or one a
        nested request unit's cancellation ended (a session connect, a Decision question). Its
        error is the RetryCancelled that stopped it, which names the attempt of the unit that
        raised it. Its usage is that of the questions a Decision answered before, which their own
        records also hold (decision 37). An attempt whose reply had arrived is recorded by
        _attempt_finished, with its usage."""
        with self._attempt_scope(generation, request, attempt):
            self._trace.capture(
                TraceEvent.MODEL_ATTEMPT_FINISHED,
                {
                    "kind": "generation",
                    "outcome": "cancelled",
                    "error": repr(exc)[:500],
                    "usage": getattr(exc, "partial_usage", None),
                },
            )

    def abort(self, generation: int):
        super().abort(generation)
        aborted = self._cancel(generation)
        # A generation aborted before its worker started generate() has no decision yet.
        decision = aborted[0].decision if aborted else None
        with trace_scope(phase="abort", decision=decision, generation=generation):
            self._runtime.abort()

    def close(self):
        """Cancel the generations that have not ended, and discard them: one the episode ended
        on gets no AcceptResult or DiscardResult after Termination."""
        for generation in self._cancel():
            self.discard(generation.number)
        try:
            with trace_scope(phase="close"):
                self._runtime.close()
            self._execution_thread.stop()
        finally:
            super().close()
