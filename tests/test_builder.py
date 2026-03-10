import pytest

from headroom.builder import DropSlotCompactor, PromptBuilder, TruncateCompactor

@pytest.mark.parametrize(
    "pb,expected_prompt",
    [
        pytest.param(
            PromptBuilder().context("hello"), 
            "hello",
            id="hello_context_only",
        ),
        pytest.param(
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
5""",
            id="section_ordering"
        )
    ]
)
def test_basic(pb: PromptBuilder, expected_prompt: str):
    assert pb.build() == expected_prompt

@pytest.mark.parametrize(
    "pb,expected_prompt",
    [
        pytest.param(
            PromptBuilder()
                .system("You are a friendly assistant")
                .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
            id="simple_drop_large_context"
        ),
        pytest.param(
            PromptBuilder()
                .system("You are a friendly assistant")
                .context("a" * ((1_000 * 4) + 100))
                .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
            id="drop_all_large_context"
        ),
        pytest.param(
            PromptBuilder()
                .system("You are a friendly assistant")
                .instructions("Please summarize the following")
                .context("a" * ((1_000 * 4) + 100)),
            """You are a friendly assistant
Please summarize the following""",
            id="drop_context_before_instructions"
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=20,
                compactors=(TruncateCompactor(max_chars=5), DropSlotCompactor())
            )
            .system("You are a friendly assistant")
            .context("a" * 40)
            .context("a" * 40),
            """Yo...
aa...
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa""",
            id="truncate_all_before_drop"
        )
    ]
)
def test_compaction(pb: PromptBuilder, expected_prompt: str):
    assert pb.build() == expected_prompt

def test_char_estimate_over_budget():
    pb = PromptBuilder(disable_compaction=True).system("You are a friendly assistant").context("a" * ((1_000 * 4) + 100))

    with pytest.raises(ValueError):
        pb.build()
