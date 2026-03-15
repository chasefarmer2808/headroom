import pytest
import tiktoken

from headroom.counter import (
    CharEstimateCounter,
    TikTokenCounter,
    TokenCounter,
    get_counter,
)


def test_char_estimate_known_str():
    assert 4 == CharEstimateCounter().count_tokens("how are you today?")


@pytest.mark.parametrize(
    "encoding,expected_counter",
    [
        ("unknown", CharEstimateCounter()),
        ("o200k_base", TikTokenCounter(tiktoken.get_encoding("o200k_base"))),
    ],
)
def test_get_counter(encoding: str, expected_counter: TokenCounter):
    test_prompt = "this is a test prompt"
    counter = get_counter(encoding)
    assert type(counter) is type(expected_counter)
    assert counter.count_tokens(test_prompt) == expected_counter.count_tokens(
        test_prompt
    )
