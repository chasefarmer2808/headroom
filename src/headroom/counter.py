from typing import NamedTuple, Protocol


class ModelSpec(NamedTuple):
    context_window: int
    encoder: str


MODEL_REGISTRY: dict[str, ModelSpec] = {"gpt-4o": ModelSpec(128_000, "o200k_base")}


class TokenCounter(Protocol):
    def count_tokens(self, prompt: str) -> int: ...


class CharEstimateCounter:
    def count_tokens(self, prompt: str) -> int:
        return len(prompt) // 4


def get_counter(model_name: str) -> TokenCounter:
    spec = MODEL_REGISTRY.get(model_name)

    if not spec:
        return CharEstimateCounter()
