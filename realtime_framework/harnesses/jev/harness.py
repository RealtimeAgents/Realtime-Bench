"""Choice harness: enumerate actions and select through the registered Decisions encoder.

Text-only clients receive structured control state; multimodal clients additionally receive
its marked screenshot. Candidate and text-composition policies are shared by both. History is
replayed canonically, like every other model, unless the context policy gives an action digest.
"""

from __future__ import annotations

import json
from dataclasses import replace

from realtime_framework.entity.context_policy import phase1_context_policy
from realtime_framework.entity.request import Candidate, ChoiceRequirements, SelectionMaterial
from realtime_framework.harnesses.default.context import RealtimeRuntime
from realtime_framework.harnesses.default.harness import DefaultHarness
from realtime_framework.harnesses.default.prompts import GAME_SYSTEM_PROMPT
from realtime_framework.harnesses.default.request import build_request
from realtime_framework.harnesses.jev.candidates import CANDIDATE_POLICY, observation_candidates
from realtime_framework.harnesses.jev.game_candidates import (
    GAME_CANDIDATE_POLICY,
    add_game_candidates,
)
from realtime_framework.harnesses.jev.prompts import (
    JEV_PROFILE,
    MULTIMODAL_CHOICE_PROFILE,
    MULTIMODAL_GAME_CHOICE_PROFILE,
)
from realtime_framework.providers.decisions.questions import (
    GAME_INSTRUCTIONS,
    GUI_INSTRUCTIONS,
    TEXT_COMPOSITION_POLICY,
)
from realtime_framework.providers.registry import adapter_class
from realtime_framework.sandbox.observation.interactables import format_interactables
from realtime_framework.sandbox.observation.som import OBSERVATION_INTERACTABLES_HEADER


class JevHarness(DefaultHarness):
    interaction_profile = JEV_PROFILE

    def _setup_backend(self):
        self.driver = adapter_class(self.config.adapter_type)(self.config)
        self.driver.bind_trace(self._trace)
        self.interaction_profile = (
            MULTIMODAL_CHOICE_PROFILE
            if self.driver.capabilities.observation_modality == "screenshot"
            else JEV_PROFILE
        )
        self._context_policy = phase1_context_policy().for_path(
            self.config.adapter_type, self.config.provider_options
        )
        self._runtime = RealtimeRuntime(self.driver, self._context_policy, trace=self._trace)

    def initialize(self, task_instruction, initial_observation):
        # The pre-run checks (providers/preflight.py) admit only the wall clock and, for
        # text-only clients, web tasks.
        scenario = getattr(self, "system_prompt_override", None)
        self._game_choices = scenario == GAME_SYSTEM_PROMPT
        self.interaction_profile = (
            MULTIMODAL_GAME_CHOICE_PROFILE if self._game_choices else
            MULTIMODAL_CHOICE_PROFILE
            if self.driver.capabilities.observation_modality == "screenshot" else JEV_PROFILE
        )
        super().initialize(task_instruction, initial_observation)

    def _prepare_observations(self, observations, tick, feedback=""):
        prepared, executions = super()._prepare_observations(observations, tick, feedback)
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
            # Only the current observation's candidates are executable.
            self._choice_state = state
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
        return prepared, executions

    def _build_request(self, turns, ledger, call):
        # The one observation this decision prepared (the runtime takes exactly one per
        # generation); its candidates are the answer options and the rest of its state is
        # material for the selection flow only.
        state = self._choice_state
        choice = ChoiceRequirements(
            instructions=GAME_INSTRUCTIONS if state["game_choices"] else GUI_INSTRUCTIONS,
            candidates=tuple(
                (key, Candidate(value["description"], value["action"], value.get("audit")))
                for key, value in state["candidates"].items()
            ),
            candidate_policy=state["candidate_policy"],
        )
        selection = SelectionMaterial(
            observation_id=state["observation_id"],
            game_choices=state["game_choices"],
            focused_input=state["focused_input"],
            text_composition=state["text_composition"],
        )
        return build_request(
            self._context_policy,
            self._system_block,
            turns,
            ledger,
            replace(self._answer, choice=choice),
            call,
            selection,
        )
