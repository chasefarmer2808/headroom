from typing import get_args

import pytest
import tiktoken

from headroom.counter import (
    MODEL_REGISTRY,
    CharEstimateCounter,
    ModelName,
    TikTokenCounter,
    TokenCounter,
    get_counter,
)


def test_char_estimate_known_str():
    assert 4 == CharEstimateCounter().count_tokens("how are you today?")


@pytest.mark.parametrize(
    "encoding,expected_counter",
    [
        ("unknown", None),
        ("o200k_base", TikTokenCounter(tiktoken.get_encoding("o200k_base"))),
    ],
)
def test_get_counter(encoding: str, expected_counter: TokenCounter | None):
    test_prompt = "this is a test prompt"
    counter = get_counter(encoding)
    assert type(counter) is type(expected_counter)
    if expected_counter:
        assert counter.count_tokens(test_prompt) == expected_counter.count_tokens(
            test_prompt
        )
    else:
        assert counter is None


def test_all_model_names_have_registry_entries():
    for name in get_args(ModelName):
        assert name in MODEL_REGISTRY, f"Missing registry entry for {name!r}"
