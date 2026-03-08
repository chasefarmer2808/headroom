from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple, Protocol, TypedDict, runtime_checkable

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

class PromptBuilder:
    def __init__(self):
        self._slots: PromptSlots = {
            "system": [],
            "instructions": [],
            "context": [],
            "history": [],
            "user": [],
        }
    
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
        return "\n".join(
            f.content.to_prompt() if isinstance(f.content, Promptable) else f.content
            for slot in self._slots.values()
            for f in slot
        )

def main():
    builder = PromptBuilder().context("You are a helpful assistant")
    print(builder.build())


if __name__ == "__main__":
    main()
