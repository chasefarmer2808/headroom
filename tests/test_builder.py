import pytest

from headroom.builder import DropSlotCompactor, PromptBuilder, TruncateCompactor

@pytest.mark.parametrize(
    "pb,expected_prompt",
    [
        (
            PromptBuilder().context("hello"), 
            "hello",
        ),
        (
            PromptBuilder()
                .system("1")
                .instructions("2")
                .context("3")
                .history("4")
                .user("5"),
            """1
2
3
4
5"""
        )
    ]
)
def test_basic(pb: PromptBuilder, expected_prompt: str):
    assert pb.build() == expected_prompt

@pytest.mark.parametrize(
    "pb,expected_prompt",
    [
        (
            PromptBuilder()
                .system("You are a friendly assistant")
                .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
        ),
        (
            PromptBuilder()
                .system("You are a friendly assistant")
                .context("a" * ((1_000 * 4) + 100))
                .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
        ),
        (
            PromptBuilder(
                max_tokens=20,
                compactors=(TruncateCompactor(max_chars=5), DropSlotCompactor())
            )
            .system("You are a friendly assistant")
            .context("a" * 40)
            .context("a" * 40),
            """Yo...
aa...
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"""
        )
    ]
)
def test_compaction(pb: PromptBuilder, expected_prompt: str):
    assert pb.build() == expected_prompt

def test_char_estimate_over_budget():
    pb = PromptBuilder(disable_compaction=True).system("You are a friendly assistant").context("a" * ((1_000 * 4) + 100))

    with pytest.raises(ValueError):
        pb.build()
