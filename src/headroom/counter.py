from typing import Protocol


class TokenCounter(Protocol):
    def count_tokens(self, prompt: str) -> int: ...


class CharEstimateCounter:
    def count_tokens(self, prompt: str) -> int:
        return len(prompt) // 4
