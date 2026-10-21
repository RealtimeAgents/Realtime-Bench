from __future__ import annotations

# Canonical request
"""Everything one decision sends to the model, in a form no provider API owns.

Content blocks keep their number, time and source; the role and placement each one is given are
decided when the request is generated, from the context policy, and travel on the request items.
Encoders only map that intent onto their protocol.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

from realtime_framework.entity.records import _jsonable

if TYPE_CHECKING:  # entity.context_policy imports entity.model, which imports this module
    from realtime_framework.entity.context_policy import PathPolicy

BlockKind = Literal[
    "system_instruction",
    "frame",
    "model_output",
    "execution_feedback",
    "action_ledger",
    "tool_exchange",
]
Source = Literal["environment", "agent", "model"]
# Neutral roles; no provider role names.
Role = Literal["instruction", "input", "model"]
Position = Literal["head", "after_head", "in_order", "frame_prefix", "digest"]


@dataclass(frozen=True)
class BlockTime:
    wall_ns: int | None = None  # agent wall clock when written (output, feedback, ledger, system)
    episode_s: float | None = None  # frame: Observation.timestamp
    tick: float | None = None  # frame: Observation.tick_number


@dataclass(frozen=True)
class FrameImage:
    data: bytes  # the model-sized screenshot, SoM-marked when inspected
    mime_type: str
    detail: str


@dataclass(frozen=True)
class ToolExchange:
    """One live tool round, in provider order."""

    calls: tuple[dict, ...]  # {"id", "name", "arguments"}
    receipts: tuple[dict, ...]  # the acknowledged FunctionResponse dumps
    text_after: str = ""  # model text that followed the call


@dataclass(frozen=True)
class ContentBlock:
    # "system", "ledger", or "d{decision}.frame|prefix|output|feedback|tools"
    id: str
    kind: BlockKind
    source: Source
    time: BlockTime
    # frame: prompt with its SoM block (folded: the placeholder); output: text or action JSON;
    # feedback: the execution result; ledger: its JSON Lines; system: the instruction;
    # tools: the exchanges as text when the policy replays them as a transcript.
    text: str = ""
    image: FrameImage | None = None  # unfolded frames of screenshot observations only
    folded: bool = False
    # frame: the existing frame number; digest feedback an older frame opened with: that frame's
    observation_id: str | None = None
    based_on: str | None = None  # output and tools: observation_id of the frame answered
    attached_to: str | None = None  # feedback: the output it reports, or the frame it prefixes
    reasoning: str = ""  # output: set only when the policy replays reasoning
    initial_text: str = ""  # tools: model text before the first call
    exchanges: tuple[ToolExchange, ...] = ()
    revision: int | None = None  # ledger: archived action count it covers


@dataclass(frozen=True)
class Intent:
    role: Role
    position: Position


@dataclass(frozen=True)
class RequestItem:
    block: ContentBlock
    intent: Intent


@dataclass(frozen=True)
class Candidate:
    description: str
    action: dict
    audit: dict | None = None


@dataclass(frozen=True)
class ChoiceRequirements:
    """The root question of one Decision; the selection flow derives the questions after it."""

    instructions: str
    candidates: tuple[tuple[str, Candidate], ...]  # (choice id, candidate) in criteria order
    candidate_policy: str


@dataclass(frozen=True)
class AnswerRequirements:
    action_channel: Literal["text_json", "function_call", "choice"]
    # InteractionProfile.name; its action space and output contract are in the system block.
    format_profile: str
    function_declarations: tuple[dict, ...] = ()  # function_call channel
    choice: ChoiceRequirements | None = None


@dataclass(frozen=True)
class SelectionMaterial:
    """Current-frame input of the Decision selection flow; never rendered, never stored in
    history."""

    observation_id: str
    game_choices: bool
    focused_input: dict | None  # also part of the visible frame JSON
    text_composition: dict | None  # the composition inventory


@dataclass(frozen=True)
class CallParameters:
    model: str
    reasoning_effort: str | None
    output_token_cap: int | None  # the decision's provider output cap


@dataclass(frozen=True)
class CanonicalRequest:
    # The context policy the request was generated under; encoders read its fixed texts from
    # here.
    policy: PathPolicy
    history_form: Literal["turns", "action_digest"]
    items: tuple[RequestItem, ...]  # in the order they are sent
    answer: AnswerRequirements
    call: CallParameters
    selection: SelectionMaterial | None = None

    def to_record(self) -> dict:
        """A JSON form for audit records; image bytes become their length and SHA-256.

        The policy is named by version and API; the effective run config records its values.
        """
        values = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "policy"}
        return {"policy_version": self.policy.version, "api": self.policy.api, **_jsonable(values)}
