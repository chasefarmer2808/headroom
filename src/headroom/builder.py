from __future__ import annotations

import copy
import logging
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from .counter import MODEL_REGISTRY, CharEstimateCounter, TokenCounter

logger = logging.getLogger(__name__)


@runtime_checkable
class Promptable(Protocol):
    def to_prompt(self) -> str: ...


@runtime_checkable
class Compactable(Protocol):
    def compact(self) -> Promptable: ...


type _Promptable = Promptable | str


class ExhaustionPolicy(Enum):
    WARN = auto()
    RAISE = auto()


class Importance(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class Fragment(NamedTuple):
    content: _Promptable
    importance: Importance


class CompactionResult(NamedTuple):
    slot_name: str
    index: int
    fragment: Fragment
    operation: Literal["replace", "delete"]


class PromptSlots(TypedDict):
    system: list[Fragment]
    instructions: list[Fragment]
    context: list[Fragment]
    history: list[Fragment]
    user: list[Fragment]


@dataclass
class CompactionEvent:
    compactor_name: str
    slot: str
    fragment_type: str
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class BuildResult:
    prompt: str
    tokens_used: int
    token_budget: int
    compaction_events: tuple[CompactionEvent, ...]

    # TODO add utilization and tokens_remaining properties


class Compactor(Protocol):
    def apply(self, fragment: Fragment) -> Fragment | Literal["drop"] | None: ...


class TruncateCompactor:
    def __init__(self, max_chars: int = 500):
        self._max_chars = max_chars

    def apply(self, fragment: Fragment) -> Fragment | Literal["drop"] | None:
        if (
            isinstance(fragment.content, str)
            and len(fragment.content) > self._max_chars
        ):
            return Fragment(
                fragment.content[: self._max_chars - len("...")] + "...",
                fragment.importance,
            )


class InlineCompactor:
    def apply(self, fragment: Fragment) -> Fragment | Literal["drop"] | None:
        if isinstance(fragment.content, Compactable):
            return Fragment(fragment.content.compact(), fragment.importance)


class DropFragCompactor:
    """
    Applies compaction by completely removing the
    lowest important fragment from the lowest important slot.
    Intended to be used as a last resort when other
    compaction methods aren't viable or available.
    """

    def apply(self, _: Fragment) -> Fragment | Literal["drop"] | None:
        return "drop"


class PromptBuilder:
    def __init__(
        self,
        model_name: str
        | None = None,  # TODO should be type safe to the keys of the model registry.
        max_tokens: int | None = None,
        compactors: tuple[Compactor, ...] = (
            InlineCompactor(),
            TruncateCompactor(),
            DropFragCompactor(),
        ),
        disable_compaction: bool = False,
        exhaustion_policy: ExhaustionPolicy = ExhaustionPolicy.WARN,
    ):
        self._model_name = model_name
        self._max_tokens = max_tokens or 1_000
        self._token_counter: TokenCounter = CharEstimateCounter()
        self._compactors = compactors
        self._slots: PromptSlots = {
            "system": [],
            "instructions": [],
            "context": [],
            "history": [],
            "user": [],
        }
        self._slot_order: tuple[str, ...] = ("history", "context", "instructions")
        self._disable_compaction = disable_compaction
        self._exhaustion_policy = exhaustion_policy

        # Initialize the token budget based on model name or max tokens
        if self._model_name:
            # set max tokens for that model
            model_spec = MODEL_REGISTRY.get(self._model_name)
            if model_spec:
                self._max_tokens = model_spec.context_window

        if max_tokens:
            self._max_tokens = max_tokens

    def system(self, p: _Promptable, importance=Importance.CRITICAL) -> PromptBuilder:
        self._slots["system"].append(Fragment(p, importance))
        return self

    def instructions(self, p: _Promptable, importance=Importance.HIGH) -> PromptBuilder:
        self._slots["instructions"].append(Fragment(p, importance))
        return self

    def context(self, p: _Promptable, importance=Importance.NORMAL) -> PromptBuilder:
        self._slots["context"].append(Fragment(p, importance))
        return self

    def history(self, p: _Promptable, importance=Importance.LOW) -> PromptBuilder:
        self._slots["history"].append(Fragment(p, importance))
        return self

    def user(self, p: _Promptable, importance=Importance.CRITICAL) -> PromptBuilder:
        self._slots["user"].append(Fragment(p, importance))
        return self

    def build(self) -> BuildResult:
        compaction_events: list[CompactionEvent] = []
        slots = copy.deepcopy(self._slots)
        prompt_str = self._render(slots)
        curr_count = self._token_counter.count_tokens(prompt_str)

        if curr_count <= self._max_tokens:
            return BuildResult(
                prompt_str,
                curr_count,
                self._max_tokens,
                tuple(compaction_events),
            )

        if self._disable_compaction:
            raise ValueError("Prompt exceeds max token budget")
        # Apply exhaustive sequential compaction
        for compactor in self._compactors:
            for slot_name, i, compacted_frag, op in self._compact_next(compactor):
                if op == "replace":
                    slots[slot_name][i] = compacted_frag
                elif op == "delete":
                    slots[slot_name].remove(compacted_frag)

                prompt_str = self._render(slots)
                count_after_compaction = self._token_counter.count_tokens(prompt_str)

                compaction_events.append(
                    CompactionEvent(
                        type(compactor),
                        slot_name,
                        type(compacted_frag.content),
                        curr_count,
                        count_after_compaction,
                    )
                )

                if count_after_compaction <= self._max_tokens:
                    return BuildResult(
                        prompt_str,
                        count_after_compaction,
                        self._max_tokens,
                        tuple(compaction_events),
                    )

                curr_count = count_after_compaction

        if count_after_compaction > self._max_tokens:
            if self._exhaustion_policy == ExhaustionPolicy.WARN:
                logger.warning(
                    "Prompt is still over budget after all compactions were exhausted. "
                    "Token count: %d, budget: %d.",
                    count_after_compaction,
                    self._max_tokens,
                )
            elif self._exhaustion_policy == ExhaustionPolicy.RAISE:
                raise ValueError(
                    "Prompt is still over budget after all compactions were exhausted."
                )

        return BuildResult(
            prompt_str,
            count_after_compaction,
            self._max_tokens,
            tuple(compaction_events),
        )

    def _compact_next(self, compactor: Compactor) -> Generator[CompactionResult]:
        for slot_name in self._slot_order:
            frags = self._slots.get(slot_name, [])
            sorted_frags = sorted(
                [
                    (i, f)
                    for i, f in enumerate(frags)
                    if f.importance != Importance.CRITICAL
                ],
                key=lambda pair: pair[1].importance.value,
            )
            for i, frag in sorted_frags:
                result = compactor.apply(frag)
                if result is None:
                    continue
                elif result == "drop":
                    yield CompactionResult(slot_name, i, frag, "delete")
                else:
                    yield CompactionResult(slot_name, i, result, "replace")

    def _render(self, slots: PromptSlots) -> str:
        return "\n".join(
            f.content.to_prompt() if isinstance(f.content, Promptable) else f.content
            for slot in slots.values()
            for f in slot
        )


def main():
    builder = PromptBuilder().context("You are a helpful assistant")
    print(builder.build())


if __name__ == "__main__":
    main()
