from __future__ import annotations

from typing import Protocol, runtime_checkable

@runtime_checkable
class Promptable(Protocol):
    def to_prompt(self) -> str:
        ...

type _Promptable = Promptable | str

class PromptBuilder:
    def __init__(self):
        self._promptables: list[_Promptable] = []
    
    def context(self, context: _Promptable) -> PromptBuilder:
        self._promptables.append(context)
        return self
    
    def build(self) -> str:
        all_prompts = [promptable.to_prompt() if isinstance(promptable, Promptable) else promptable for promptable in self._promptables]
        return "\n".join(all_prompts)

def main():
    builder = PromptBuilder().context("You are a helpful assistant")
    print(builder.build())


if __name__ == "__main__":
    main()
