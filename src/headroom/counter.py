from typing import Literal, NamedTuple, Protocol

import tiktoken
from tiktoken import Encoding


class ModelSpec(NamedTuple):
    context_window: int
    encoder: str


ModelName = Literal["gpt-4o"]

MODEL_REGISTRY: dict[ModelName, ModelSpec] = {
    "gpt-4o": ModelSpec(128_000, "o200k_base")
}


class TokenCounter(Protocol):
    def count_tokens(self, prompt: str) -> int: ...


class CharEstimateCounter:
    def count_tokens(self, prompt: str) -> int:
        return len(prompt) // 4


class TikTokenCounter:
    def __init__(self, encoding: Encoding):
        self._encoding = encoding

    def count_tokens(self, prompt: str) -> int:
        return len(self._encoding.encode(prompt))


def get_counter(encoding_name: str) -> TokenCounter:
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return TikTokenCounter(encoding)
    except ValueError:
        return CharEstimateCounter()
