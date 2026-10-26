"""The model-visible history of a turn-based episode, without I/O.

It holds the retained turns, the action record of each accepted decision, the archived-action
ledger the records project, and the feedback on decisions of which nothing was accepted, which
no turn carries yet. Everything that changes it is a change written through ``History.write``:
before a generation the policy archive and screenshot fold, while its request is fitted a
pressure archive or a ledger omission (the runtime, harnesses/default/context.py); a decision
accepted, its outcome committed, or the feedback on a decision of which nothing was accepted
(the agent, harnesses/default/harness.py). An action record names what its decision asked for
once it is accepted, and what ran and how once its outcome is committed.

The runtime's writes and an accepted decision's run on the agent's execution thread, where a
generation that was abandoned but still runs reads the history; a commit's runs on the shell's
thread and changes only its own decision's record and turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from realtime_framework.entity.context_policy import PathPolicy
from realtime_framework.entity.model import CanonicalTurn
from realtime_framework.entity.request import ContentBlock
from realtime_framework.harnesses.default.request import ledger_block


@dataclass(frozen=True)
class Retain:
    """The policy archive and screenshot fold before a generation, over the retained turns and
    the generation's ``incoming`` ones, which the fold reaches too."""

    incoming: tuple[CanonicalTurn, ...]
    wall_ns: int  # when the ledger grows, if it does


@dataclass(frozen=True)
class ArchiveBlock:
    """A pressure archive: the earliest block of retained turns leaves for the ledger."""

    wall_ns: int  # when the ledger grows, if it does


@dataclass(frozen=True)
class OmitBlock:
    """The next block of the earliest archived actions leaves the ledger."""


@dataclass(frozen=True)
class Accept:
    """An accepted decision's turn and action record join the history, and the feedback no turn
    carried as its generation started (``unattached``) is carried by its turn."""

    decision: int
    turn: CanonicalTurn
    record: dict  # what the decision asked for; numbered as it is written
    unattached: str


@dataclass(frozen=True)
class Commit:
    """An accepted decision's outcome: what ran and how, on its action record, and its feedback,
    on its turn while the turn is retained."""

    decision: int
    outcome: dict
    feedback: str
    wall_ns: int


@dataclass(frozen=True)
class Unanswered:
    """The feedback on a decision of which nothing was accepted, which opens the next frame."""

    feedback: str


@dataclass(frozen=True)
class Retained:
    """What a Retain archived and folded, and how many actions the ledger gained."""

    archived: tuple[CanonicalTurn, ...]
    folded: tuple[CanonicalTurn, ...]
    archived_actions: int

    @property
    def changed(self) -> bool:
        """Whether turns the history already held changed."""
        return bool(self.archived or self.folded)


def _mark_oldest_active_visuals(turns: list[dict], count: int) -> int:
    marked = 0
    for turn in turns:
        if turn.get("visual_folded"):
            continue
        turn["visual_folded"] = True
        marked += 1
        if marked >= count:
            break
    return marked


def apply_active_visual_budget(turns: list[dict], budget: int, block_size: int) -> int:
    """Fold old screenshots in blocks without making any text-retention decision."""
    active = sum(not turn.get("visual_folded") for turn in turns)
    folded = 0
    while active > budget:
        count = min(block_size, active - 1)
        folded += _mark_oldest_active_visuals(turns, count)
        active -= count
    return folded


def executed_action_ledger(actions, limit: int) -> str:
    """The first ``limit`` action records as compact deterministic JSON Lines: what each asked
    for, and what ran and how once its outcome is committed."""
    lines = []
    for item in actions[:limit]:
        record = {
            "decision": item["tick"],
            "observed_at_tick": item["observation_tick"],
            "requested": item["requested"],
            "actual": item.get("executed", "?"),
            "status": item.get("result", "pending"),
        }
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines)


class History:
    """One episode's history under ``policy``; ``write`` is the only way it changes."""

    def __init__(self, policy: PathPolicy):
        self._policy = policy
        self._turns: list[CanonicalTurn] = []
        self._actions: list[dict] = []
        self._index: dict[int, int] = {}  # each accepted decision's action record, by decision
        self.archived = 0  # the archived actions, those left out of the ledger included
        # The earliest archived actions the ledger leaves out: a request that did not fit with
        # every turn archived dropped them. As archived turns do, they stay out for the rest of
        # the episode, though a later request would fit with them, so that the history the model
        # sees does not change back.
        self.omitted = 0
        self.ledger_wall_ns: int | None = None  # when the ledger last grew
        self.unattached = ""

    @property
    def turns(self) -> tuple[CanonicalTurn, ...]:
        return tuple(self._turns)

    @property
    def actions(self) -> tuple[dict, ...]:
        """Each accepted decision's action record, in the order accepted."""
        return tuple(self._actions)

    def index(self, decision: int) -> int | None:
        """The position of an accepted decision's action record."""
        return self._index.get(decision)

    def action(self, decision: int) -> dict | None:
        """An accepted decision's action record."""
        index = self._index.get(decision)
        return None if index is None else self._actions[index]

    def ledger_block(self, omitted: int | None = None) -> ContentBlock | None:
        """The ledger as requests carry it: one line per archived action, the ``omitted``
        earliest ones (by default the ones left out so far) giving way to a note that counts
        them."""
        ledger = executed_action_ledger(self._actions, self.archived)
        omitted = self.omitted if omitted is None else omitted
        if omitted:
            kept = ledger.split("\n")[omitted:]
            note = self._policy.texts.ledger_omission.format(count=omitted)
            ledger = "\n".join([note, *kept])
        return ledger_block(ledger, self.archived, self.ledger_wall_ns)

    def write(self, change):
        """Apply one change; a Retain returns what it Retained, an ArchiveBlock the turns it
        archived and the actions the ledger gained, an OmitBlock how many actions it left out,
        none once every one is out."""
        match change:
            case Retain():
                return self._retain(change)
            case ArchiveBlock(wall_ns):
                archived = tuple(self._turns[: self._policy.turns.block])
                del self._turns[: len(archived)]
                return archived, self._archive(archived, wall_ns)
            case OmitBlock():
                count = min(self._policy.turns.block, self.archived - self.omitted)
                self.omitted += count
                return count
            case Accept(decision, turn, record, unattached):
                self._turns.append(turn)
                self._index[decision] = len(self._actions)
                self._actions.append({"tick": len(self._actions) + 1, **record})
                if self.unattached == unattached:
                    self.unattached = ""
            case Commit(decision, outcome, feedback, wall_ns):
                self._actions[self._index[decision]].update(outcome)
                turns = [turn for turn in self._turns if turn.decision == decision]
                if turns:
                    turns[-1].execution_feedback = feedback
                    turns[-1].feedback_wall_ns = wall_ns
            case Unanswered(feedback):
                self.unattached = "\n".join(filter(None, (self.unattached, feedback)))
            case _:
                raise TypeError(f"not a history change: {change!r}")

    def _retain(self, change: Retain) -> Retained:
        window, block_size = self._policy.turns.keep, self._policy.turns.block
        archive_count = 0
        while len(self._turns) - archive_count + len(change.incoming) > window:
            removable = len(self._turns) - archive_count
            if removable <= 0:
                break
            archive_count += min(block_size, removable)
        archived = tuple(self._turns[:archive_count])
        del self._turns[:archive_count]
        archived_actions = self._archive(archived, change.wall_ns)
        candidates = [*self._turns, *change.incoming]
        previously_folded = {id(turn): turn.visual_folded for turn in candidates}
        records = [turn.retention_record() for turn in candidates]
        apply_active_visual_budget(records, self._policy.visuals.keep, self._policy.visuals.block)
        for turn, record in zip(candidates, records):
            if record.get("visual_folded"):
                turn.fold_visual()
        folded = tuple(
            turn for turn in candidates if turn.visual_folded and not previously_folded[id(turn)]
        )
        return Retained(archived, folded, archived_actions)

    def _archive(self, turns: tuple[CanonicalTurn, ...], wall_ns: int) -> int:
        """Archive the accepted decisions of ``turns`` into the ledger; returns how many actions
        it gained."""
        before = self.archived
        if turns:
            self.archived = max(before, max(self._index[turn.decision] for turn in turns) + 1)
        if self.archived > before:
            self.ledger_wall_ns = wall_ns
        return self.archived - before
