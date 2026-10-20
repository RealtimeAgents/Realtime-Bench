"""Choice harness: enumerate actions and select through a stateless ChoiceClient.

Text-only clients receive structured control state; multimodal clients additionally receive
its marked screenshot. Candidate and text-composition policies are shared by both. History is
replayed canonically, like every other model, unless the client declares an action digest.
"""

from __future__ import annotations

import base64
import json

from realtime_framework.entity.context_policy import phase1_context_policy
from realtime_framework.harnesses.default.context import RealtimeRuntime
from realtime_framework.harnesses.default.harness import DefaultHarness
from realtime_framework.harnesses.default.prompts import (
    GAME_SYSTEM_PROMPT,
    REALTIME_WEB_SYSTEM_PROMPT,
)
from realtime_framework.harnesses.jev.candidates import CANDIDATE_POLICY, observation_candidates
from realtime_framework.harnesses.jev.composition import TEXT_COMPOSITION_POLICY
from realtime_framework.harnesses.jev.context import observation_messages
from realtime_framework.harnesses.jev.driver import ChoiceDecisionDriver
from realtime_framework.harnesses.jev.game_candidates import (
    GAME_CANDIDATE_POLICY,
    add_game_candidates,
)
from realtime_framework.harnesses.jev.prompts import (
    JEV_PROFILE,
    MULTIMODAL_CHOICE_PROFILE,
    MULTIMODAL_GAME_CHOICE_PROFILE,
)
from realtime_framework.providers.registry import adapter_class
from realtime_framework.sandbox.observation.interactables import format_interactables
from realtime_framework.sandbox.observation.som import OBSERVATION_INTERACTABLES_HEADER


class JevHarness(DefaultHarness):
    interaction_profile = JEV_PROFILE

    def _setup_backend(self):
        self._observation_states = {}
        client = adapter_class(self.config.adapter_type)(self.config)
        self.interaction_profile = (
            MULTIMODAL_CHOICE_PROFILE
            if client.capabilities.observation_modality == "screenshot"
            else JEV_PROFILE
        )
        self.driver = ChoiceDecisionDriver(client, self._observation_states)
        self._context_policy = phase1_context_policy().for_path(
            self.config.adapter_type, self.config.provider_options
        )
        self._runtime = RealtimeRuntime(self.driver, self._context_policy)

    def initialize(self, task_instruction, initial_observation):
        if getattr(self, "_clock_mode", None) != "wall":
            raise ValueError("Choice Decisions supports wall clock only")
        if getattr(self, "_stream_mode", "default") != "default":
            raise ValueError("Choice Decisions requires default mode")
        if getattr(self, "tool_runtime", None) is None:
            raise ValueError("Choice Decisions requires a task-bound interactables runtime")
        scenario = getattr(self, "system_prompt_override", None)
        self._game_choices = scenario == GAME_SYSTEM_PROMPT
        if self._game_choices and self.driver.capabilities.observation_modality != "screenshot":
            raise ValueError(f"Choice Decisions supports Web task evaluation only: {self.tool_runtime.task_id}")
        if scenario not in {GAME_SYSTEM_PROMPT, REALTIME_WEB_SYSTEM_PROMPT}:
            raise ValueError(f"Unsupported Choice Decisions scenario: {self.tool_runtime.task_id}")
        self.interaction_profile = (
            MULTIMODAL_GAME_CHOICE_PROFILE if self._game_choices else
            MULTIMODAL_CHOICE_PROFILE
            if self.driver.capabilities.observation_modality == "screenshot" else JEV_PROFILE
        )
        super().initialize(task_instruction, initial_observation)

    def configure_decision(self, budget):
        if budget.clock_mode != "wall":
            raise ValueError("Choice Decisions supports wall clock only")
        super().configure_decision(budget)

    def _prepare_observations(self, observations, tick, feedback=""):
        prepared, executions = super()._prepare_observations(observations, tick, feedback)
        # Only the current observation's candidates are executable.
        self._observation_states.clear()
        for observation, execution in zip(prepared, executions):
            delivered = (execution or {}).get("delivered") or {}
            marks = (execution or {}).get("marks") or {}
            active_window = next(
                (window for window in delivered.get("windows", []) if window.get("active")),
                {},
            )
            candidates, focused_input, text_composition = observation_candidates(
                self.task_instruction, marks, active_window
            )
            if self._game_choices:
                add_game_candidates(candidates, marks, delivered, observation.geometry)
            state = {
                "observation_id": observation.observation_id,
                "timestamp": observation.observation.timestamp,
                "mark_legend": (execution or {}).get("text", ""),
                "full_som": format_interactables(delivered) if delivered else "",
                "som_truncated": False,
                "ax_tree": active_window.get("ax_tree") or "",
                "ax_capture": active_window.get("ax_capture") or {},
                "focused_input": focused_input,
                "candidate_count": len(candidates),
                "candidate_policy": GAME_CANDIDATE_POLICY if self._game_choices else CANDIDATE_POLICY,
                "game_choices": self._game_choices,
                "text_composition_policy": TEXT_COMPOSITION_POLICY,
                "text_composition": text_composition,
                "candidates": candidates,
            }
            self._observation_states[observation.observation_id] = state
            # Candidates and the composition inventory are current-only Choice criteria and stay
            # in the decision audit; retained history keeps the observation state.
            visible = {
                key: value for key, value in state.items()
                if key not in {"candidates", "text_composition", "game_choices"}
            }
            observation.prompt = (
                self.interaction_profile.frame_prompt + OBSERVATION_INTERACTABLES_HEADER
                + json.dumps(visible, ensure_ascii=False)
            )
            capabilities = self.driver.capabilities
            # Canonical messages already carry each retained screenshot.
            if (
                capabilities.observation_modality == "screenshot"
                and capabilities.history_replay == "action_digest"
            ):
                encoded = base64.b64encode(observation.image_bytes).decode("ascii")
                state["images"] = [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{observation.mime_type};base64,{encoded}",
                        "detail": observation.image_detail,
                    },
                }]
        return prepared, executions

    def _messages(self, turns):
        if self.driver.capabilities.history_replay == "canonical":
            return super()._messages(turns)
        return observation_messages(self.interaction_profile.system_instruction(self), turns)
