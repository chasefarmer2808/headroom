from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple, Protocol, TypedDict, runtime_checkable, Sequence

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

class PromptSlots(TypedDict):
    system: list[Fragment]
    instructions: list[Fragment]
    context: list[Fragment]
    history: list[Fragment]
    user: list[Fragment]

class Compactor(Protocol):
    def compact(self, slots: PromptSlots) -> PromptSlots:
        ...

class DropSlotCompactor:
    """
    Applies compaction by completely removing the lowest important fragment from the lowest important slot.
    Intended to be used as a last resort when other compaction methods aren't viable or available.
    """
    def compact(self, slots: PromptSlots) -> PromptSlots:
        # Drop non-critical fragments in droppable slots
        # Slot priority in asc order is history -> context -> instructions.  System and user never touched.
        # TODO: Make this order more explicit.  It should be type safe, encoded in PromptSlots, and shouldn't have to create the tuple every time.
        for droppable_slot in ("history", "context", "instructions"):
            frags = slots.get(droppable_slot, [])
            sorted_droppable_frags = sorted(
                [f for f in frags if f.importance != Importance.CRITICAL], # Never drop critical fragments
                key=lambda f: f.importance.value,
            )
            if not sorted_droppable_frags:
                # No more frags to drop in the current category.
                continue
            
            # Take all but the first from sorted_droppable_frags
            slots[droppable_slot] = [f for f in frags if f != sorted_droppable_frags[0]]

        return slots

class PromptBuilder:
    def __init__(self, disable_compaction: bool = False):
        self._max_tokens: int = 1_000
        self._token_counter: TokenCounter = CharEstimateCounter()
        self._compactors: tuple[Compactor, ...] = (DropSlotCompactor(),)
        self._slots: PromptSlots = {
            "system": [],
            "instructions": [],
            "context": [],
            "history": [],
            "user": [],
        }
        self._disable_compaction: bool = disable_compaction
    
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

        # Apply compaction
        for compactor in self._compactors:
            compacted_slots = compactor.compact(self._slots)
            prompt_str = self._render(compacted_slots)

            if self._token_counter.count_tokens(prompt_str) > self._max_tokens:
                break

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
