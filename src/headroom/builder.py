from __future__ import annotations

from enum import Enum, auto
from typing import Generator, Literal, NamedTuple, Protocol, TypedDict, runtime_checkable, Sequence

from .counter import TokenCounter, CharEstimateCounter

@runtime_checkable
class Promptable(Protocol):
    def to_prompt(self) -> str:
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
    slot: str
    index: int
    fragment: Fragment
    operation: Literal["replace", "delete"]

class PromptSlots(TypedDict):
    system: list[Fragment]
    instructions: list[Fragment]
    context: list[Fragment]
    history: list[Fragment]
    user: list[Fragment]

# TODO consider if this protocal is too general.  It creates a scenario where many compactors will need to perform
# type checks as they enumerate over frags.  Consider compaction capability-based compactor protocols as an alternative.
class Compactor(Protocol):
    def compact_next(self, slots: PromptSlots) -> Generator[CompactionResult, None, None]:
        ...

class TruncateCompactor:
    def __init__(self, max_chars: int = 500):
        self._max_chars = max_chars
    
    def compact_next(self, slots: PromptSlots) -> Generator[CompactionResult, None, None]:
        # TODO: this should probably only apply to context, user, and history frags
        for slot_name, frags in slots.items():
            for i, frag in enumerate(frags):
                if isinstance(frag.content, str) and len(frag.content) > self._max_chars:
                    truncated_frag = Fragment(frag.content[:self._max_chars - len("...")] + "...", frag.importance)
                    yield CompactionResult(slot_name, i, truncated_frag, "replace")


# TODO rename to DropFragCompactor
class DropFragCompactor:
    """
    Applies compaction by completely removing the lowest important fragment from the lowest important slot.
    Intended to be used as a last resort when other compaction methods aren't viable or available.
    """
    def compact_next(self, slots: PromptSlots) -> Generator[CompactionResult, None, None]:
        # Drop non-critical fragments in droppable slots
        # Slot priority in asc order is history -> context -> instructions.  System and user never touched.
        # TODO: Make this order more explicit.  It should be type safe, encoded in PromptSlots, and shouldn't have to create the tuple every time.
        for droppable_slot in ("history", "context", "instructions"):
            # TODO: figure out why I can't get better type hints on frags.  Should be Fragment but is Any.
            frags = slots.get(droppable_slot, [])
            candidates = sorted(
                [f for f in frags if f.importance != Importance.CRITICAL], # Never drop critical fragments
                key=lambda f: f.importance.value,
            )
            for frag in candidates:
                yield CompactionResult(droppable_slot, frags.index(frag), frag, "delete")

class PromptBuilder:
    def __init__(self, max_tokens: int = 1_000, compactors: tuple[Compactor, ...] = (DropFragCompactor(),), disable_compaction: bool = False):
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
            for slot_name, i, compacted_frag, op in compactor.compact_next(self._slots):
                if op == "replace":
                    self._slots[slot_name][i] = compacted_frag
                elif op == "delete":
                    self._slots[slot_name].remove(compacted_frag)

                prompt_str = self._render(self._slots)

                if self._token_counter.count_tokens(prompt_str) <= self._max_tokens:
                    return prompt_str

        # TODO: what do I do when prompt_str is still out of budget?

        return prompt_str

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
