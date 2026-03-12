from __future__ import annotations

from enum import Enum, auto
from typing import Generator, Literal, NamedTuple, Protocol, TypedDict, runtime_checkable, Sequence

from .counter import TokenCounter, CharEstimateCounter

@runtime_checkable
class Promptable(Protocol):
    def to_prompt(self) -> str:
        ...

@runtime_checkable
class Compactable(Protocol):
    def compact(self) -> Promptable:
        ...

type _Promptable = Promptable | str

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

class Compactor(Protocol):
    def apply(self, fragment: Fragment) -> Fragment | Literal["drop"] | None:
        ...

class TruncateCompactor:
    def __init__(self, max_chars: int = 500):
        self._max_chars = max_chars
    
    def apply(self, fragment: Fragment) -> Fragment | Literal["drop"] | None:
        if isinstance(fragment.content, str) and len(fragment.content) > self._max_chars:
            return Fragment(fragment.content[:self._max_chars - len("...")] + "...", fragment.importance)

class InlineCompactor:
    def apply(self, fragment: Fragment) -> Fragment | Literal["drop"] | None:
        if isinstance(fragment.content, Compactable):
            return Fragment(fragment.content.compact(), fragment.importance)

class DropFragCompactor:
    """
    Applies compaction by completely removing the lowest important fragment from the lowest important slot.
    Intended to be used as a last resort when other compaction methods aren't viable or available.
    """
    def apply(self, _: Fragment) -> Fragment | Literal["drop"] | None:
        return "drop"

class PromptBuilder:
    def __init__(
            self, 
            max_tokens: int = 1_000, 
            compactors: tuple[Compactor, ...] = (InlineCompactor(), TruncateCompactor(), DropFragCompactor(),), 
            disable_compaction: bool = False
        ):
        self._max_tokens = max_tokens
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
    
    def build(self) -> str:
        # TODO deep copy slots
        prompt_str = self._render(self._slots)

        if self._token_counter.count_tokens(prompt_str) <= self._max_tokens:
            return prompt_str
        
        if self._disable_compaction:
            raise ValueError("Prompt exceeds max token budget")
        # Apply exhaustive sequential compaction
        for compactor in self._compactors:
            for slot_name, i, compacted_frag, op in self._compact_next(compactor):
                if op == "replace":
                    self._slots[slot_name][i] = compacted_frag
                elif op == "delete":
                    self._slots[slot_name].remove(compacted_frag)

                prompt_str = self._render(self._slots)

                if self._token_counter.count_tokens(prompt_str) <= self._max_tokens:
                    return prompt_str

        # TODO: what do I do when prompt_str is still out of budget?

        return prompt_str

    def _compact_next(self, compactor: Compactor) -> Generator[CompactionResult, None, None]:
        for slot_name in self._slot_order:
            frags = self._slots.get(slot_name, [])
            sorted_frags = sorted(
                [(i, f) for i, f in enumerate(frags) if f.importance != Importance.CRITICAL],
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
