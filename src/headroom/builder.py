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

# TODO consider if this protocal is too general.  It creates a scenario where many compactors will need to perform
# type checks as they enumerate over frags.  Consider compaction capability-based compactor protocols as an alternative.
class Compactor(Protocol):
    def compact(self, slots: PromptSlots) -> PromptSlots:
        ...
    
    def can_compact(self, slots: PromptSlots) -> bool:
        ...

class TruncateCompactor:
    def __init__(self, max_chars: int = 500):
        self._max_chars = max_chars
    
    def compact(self, slots: PromptSlots) -> PromptSlots:
        # TODO: this should probably only apply to context, user, and history frags
        # TODO: this enumeration through the slots and frags to find the next target frag can probably be 
        # refactored to some kind of generator that allows me to call next().  This will also help
        # with the duplicative searching when calling can_compact and compact.  It should really just be
        # "compact if next(slots) is not None".
        for slot_name, frags in slots.items():
            for i, frag in enumerate(frags):
                if isinstance(frag.content, str) and len(frag.content) > self._max_chars:
                    truncated_frag = Fragment(frag.content[:self._max_chars - len("...")] + "...", frag.importance)
                    slots[slot_name][i] = truncated_frag
                    return slots
        
        return slots

    def can_compact(self, slots: PromptSlots) -> bool:
        for frags in slots.values():
            for frag in frags:
                if isinstance(frag.content, str) and len(frag.content) > self._max_chars:
                    return True
        
        return False


# TODO rename to DropFragCompactor
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
            # TODO: figure out why I can't get better type hints on frags.  Should be Fragment but is Any.
            frags = slots.get(droppable_slot, [])
            sorted_droppable_frags = sorted(
                [f for f in frags if f.importance != Importance.CRITICAL], # Never drop critical fragments
                key=lambda f: f.importance.value,
            )
            if not sorted_droppable_frags:
                # No more frags to drop in the current category.
                continue
            
            # Take all but the first from sorted_droppable_frags.
            # Important to check object reference equality in this filter otherwise could accidentily 
            # delete other frags with the same value.
            slots[droppable_slot] = [f for f in frags if f is not sorted_droppable_frags[0]]

        return slots

    def can_compact(self, slots: PromptSlots) -> bool:
        for droppable_slot in ("history", "context", "instructions"):
            frags = slots.get(droppable_slot, [])
            sorted_droppable_frags = sorted(
                [f for f in frags if f.importance != Importance.CRITICAL],
                key=lambda f: f.importance.value,
            )

            if len(sorted_droppable_frags) > 0:
                return True
        
        return False

class PromptBuilder:
    def __init__(self, max_tokens: int = 1_000, compactors: tuple[Compactor, ...] = (DropSlotCompactor(),), disable_compaction: bool = False):
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
            while compactor.can_compact(self._slots):
                compacted_slots = compactor.compact(self._slots)
                prompt_str = self._render(compacted_slots)

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
