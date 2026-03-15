from __future__ import annotations

import logging

import pytest

from headroom.builder import (
    DropFragCompactor,
    ExhaustionPolicy,
    Importance,
    InlineCompactor,
    PromptBuilder,
    TruncateCompactor,
)
from headroom.counter import MODEL_REGISTRY


class TestStruct:
    def __init__(self):
        self._message = "This is a very long sentence"

    def to_prompt(self) -> str:
        return self._message

    def compact(self) -> TestStruct:
        self._message = self._message.replace(" ", "")
        return self


@pytest.mark.parametrize(
    "pb,expected_prompt",
    [
        pytest.param(PromptBuilder(), "", id="empty_builder"),
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
            id="section_ordering",
        ),
        pytest.param(
            PromptBuilder().user("only user"),
            "only user",
            id="user_slot_only",
        ),
        pytest.param(
            PromptBuilder().system("sys only"),
            "sys only",
            id="system_slot_only",
        ),
        pytest.param(
            PromptBuilder().context("first").context("second").context("third"),
            "first\nsecond\nthird",
            id="multiple_fragments_same_slot",
        ),
        pytest.param(
            PromptBuilder().system("sys").user("usr"),
            "sys\nusr",
            id="system_and_user_no_middle_slots",
        ),
        pytest.param(
            PromptBuilder().context("¡Hola! 日本語 🎉"),
            "¡Hola! 日本語 🎉",
            id="unicode_content_preserved",
        ),
        pytest.param(
            PromptBuilder().context("line1\nline2\nline3"),
            "line1\nline2\nline3",
            id="multiline_fragment_preserved",
        ),
        pytest.param(
            PromptBuilder().context(TestStruct()),
            "This is a very long sentence",
            id="simple_promptable",
        ),
    ],
)
def test_basic(pb: PromptBuilder, expected_prompt: str):
    assert pb.build().prompt == expected_prompt


@pytest.mark.parametrize(
    "pb,expected_prompt",
    [
        pytest.param(
            PromptBuilder(compactors=(DropFragCompactor(),))
            .system("You are a friendly assistant")
            .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
            id="simple_drop_large_context",
        ),
        pytest.param(
            PromptBuilder(compactors=(DropFragCompactor(),))
            .system("You are a friendly assistant")
            .context("a" * ((1_000 * 2) + 100))
            .context("a" * ((1_000 * 2) + 100)),
            f"You are a friendly assistant\n{'a' * ((1_000 * 2) + 100)}",
            id="drop_one_with_same_value",
        ),
        pytest.param(
            PromptBuilder(compactors=(DropFragCompactor(),))
            .system("You are a friendly assistant")
            .context("a" * ((1_000 * 4) + 100))
            .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
            id="drop_all_large_context",
        ),
        pytest.param(
            PromptBuilder(compactors=(DropFragCompactor(),))
            .system("You are a friendly assistant")
            .instructions("Please summarize the following")
            .context("a" * ((1_000 * 4) + 100)),
            """You are a friendly assistant
Please summarize the following""",
            id="drop_context_before_instructions",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=200,
                compactors=(TruncateCompactor(max_chars=100), DropFragCompactor()),
            )
            .system("You are a friendly assistant")
            .instructions("Summarize the following pages from the book:")
            .context(f"Page 1: {('a' * 500)}")
            .context(f"Page 2: {('a' * 500)}"),
            f"""You are a friendly assistant
Summarize the following pages from the book:
Page 1: {("a" * 89)}...
Page 2: {("a" * 500)}""",
            id="truncate_first_context",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=20,
                compactors=(TruncateCompactor(max_chars=5), DropFragCompactor()),
            )
            .system("You are a friendly assistant")
            .context("a" * 40)
            .context("a" * 40),
            """You are a friendly assistant
aa...
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa""",
            id="truncate_all_before_drop",
        ),
        pytest.param(
            PromptBuilder(compactors=(DropFragCompactor(),))
            .system("You are a friendly assistant")
            .history("a" * ((1_000 * 4) + 100))
            .context("ctx"),
            "You are a friendly assistant\nctx",
            id="drop_history_before_context",
        ),
        pytest.param(
            PromptBuilder(compactors=(DropFragCompactor(),))
            .system("You are a friendly assistant")
            .history("low hist", importance=Importance.LOW)
            .history("crit hist", importance=Importance.CRITICAL)
            .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant\ncrit hist",
            id="drop_lowest_importance_history_first",
        ),
        pytest.param(
            PromptBuilder(max_tokens=6, compactors=(InlineCompactor(),))
            .system("You are a friendly assistant")
            .context(TestStruct()),
            "You are a friendly assistant\nThisisaverylongsentence",
            id="simple_inline_compactor",
        ),
    ],
)
def test_compaction(pb: PromptBuilder, expected_prompt: str):
    assert pb.build().prompt == expected_prompt


def test_custom_importance_overrides_default():
    pb = PromptBuilder(max_tokens=100_000)
    pb.history("important history", importance=Importance.HIGH)
    assert pb._slots["history"][0].importance == Importance.HIGH


def test_char_estimate_over_budget():
    pb = (
        PromptBuilder(disable_compaction=True)
        .system("You are a friendly assistant")
        .context("a" * ((1_000 * 4) + 100))
    )

    with pytest.raises(ValueError):
        pb.build()


def test_disable_compaction_within_budget_does_not_raise():
    pb = PromptBuilder(disable_compaction=True).user("short")
    assert pb.build().prompt == "short"


def test_build_idempotency():
    pb = (
        PromptBuilder()
        .system("You are a friendly assistant")
        .context("a" * ((1_000 * 4) + 100))
    )
    assert pb.build() == pb.build()


def test_exhaustion_policy_warning(caplog):
    pb = (
        PromptBuilder(max_tokens=5)
        .system("You are a friendly assistant")
        .context(TestStruct())
    )

    with caplog.at_level(logging.WARNING, logger="headroom"):
        pb.build()

    assert "Prompt is still over budget" in caplog.text


def test_exhaustion_policy_raise():
    pb = (
        PromptBuilder(max_tokens=5, exhaustion_policy=ExhaustionPolicy.RAISE)
        .system("You are a friendly assistant")
        .context(TestStruct())
    )

    with pytest.raises(ValueError) as excinfo:
        pb.build()

    assert "Prompt is still over budget" in str(excinfo.value)


def test_max_tokens_default():
    pb = PromptBuilder().system("You are a friendly assistant")
    assert pb.build().token_budget == 1_000


def test_max_tokens_safety_hatch():
    pb = PromptBuilder(max_tokens=10).system("You are a friendly assistant")

    assert pb.build().token_budget == 10


def test_token_budget_follows_model_name():
    pb = PromptBuilder(model_name="gpt-4o").system("You are a friendly assistant")

    assert pb.build().token_budget == MODEL_REGISTRY["gpt-4o"].context_window
