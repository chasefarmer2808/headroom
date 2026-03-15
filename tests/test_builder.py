from __future__ import annotations

import logging

import pytest

from headroom.builder import (
    Compactor,
    DropFragCompactor,
    ExhaustionPolicy,
    Importance,
    InlineCompactor,
    Promptable,
    PromptBuilder,
    TruncateCompactor,
)
from headroom.counter import MODEL_REGISTRY, CharEstimateCounter


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
        pytest.param(PromptBuilder(model_name="gpt-4o"), "", id="empty_builder"),
        pytest.param(
            PromptBuilder(model_name="gpt-4o").context("hello"),
            "hello",
            id="hello_context_only",
        ),
        pytest.param(
            PromptBuilder(model_name="gpt-4o")
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
            PromptBuilder(model_name="gpt-4o").user("only user"),
            "only user",
            id="user_slot_only",
        ),
        pytest.param(
            PromptBuilder(model_name="gpt-4o").system("sys only"),
            "sys only",
            id="system_slot_only",
        ),
        pytest.param(
            PromptBuilder(model_name="gpt-4o")
            .context("first")
            .context("second")
            .context("third"),
            "first\nsecond\nthird",
            id="multiple_fragments_same_slot",
        ),
        pytest.param(
            PromptBuilder(model_name="gpt-4o").system("sys").user("usr"),
            "sys\nusr",
            id="system_and_user_no_middle_slots",
        ),
        pytest.param(
            PromptBuilder(model_name="gpt-4o").context("¡Hola! 日本語 🎉"),
            "¡Hola! 日本語 🎉",
            id="unicode_content_preserved",
        ),
        pytest.param(
            PromptBuilder(model_name="gpt-4o").context("line1\nline2\nline3"),
            "line1\nline2\nline3",
            id="multiline_fragment_preserved",
        ),
        pytest.param(
            PromptBuilder(model_name="gpt-4o").context(TestStruct()),
            "This is a very long sentence",
            id="simple_promptable",
        ),
    ],
)
def test_basic(pb: PromptBuilder, expected_prompt: str):
    assert pb.build().prompt == expected_prompt


@pytest.mark.parametrize(
    "pb,expected_prompt,expected_compactions",
    [
        pytest.param(
            PromptBuilder(
                max_tokens=1_000,
                token_counter=CharEstimateCounter(),
                compactors=(DropFragCompactor(),),
            )
            .system("You are a friendly assistant")
            .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
            [(DropFragCompactor, str)],
            id="simple_drop_large_context",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=1_000,
                token_counter=CharEstimateCounter(),
                compactors=(DropFragCompactor(),),
            )
            .system("You are a friendly assistant")
            .context("a" * ((1_000 * 2) + 100))
            .context("a" * ((1_000 * 2) + 100)),
            f"You are a friendly assistant\n{'a' * ((1_000 * 2) + 100)}",
            [(DropFragCompactor, str)],
            id="drop_one_with_same_value",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=1_000,
                token_counter=CharEstimateCounter(),
                compactors=(DropFragCompactor(),),
            )
            .system("You are a friendly assistant")
            .context("a" * ((1_000 * 4) + 100))
            .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant",
            [(DropFragCompactor, str), (DropFragCompactor, str)],
            id="drop_all_large_context",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=1_000,
                token_counter=CharEstimateCounter(),
                compactors=(DropFragCompactor(),),
            )
            .system("You are a friendly assistant")
            .instructions("Please summarize the following")
            .context("a" * ((1_000 * 4) + 100)),
            """You are a friendly assistant
Please summarize the following""",
            [(DropFragCompactor, str)],
            id="drop_context_before_instructions",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=200,
                token_counter=CharEstimateCounter(),
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
            [(TruncateCompactor, str)],
            id="truncate_first_context",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=20,
                token_counter=CharEstimateCounter(),
                compactors=(TruncateCompactor(max_chars=5), DropFragCompactor()),
            )
            .system("You are a friendly assistant")
            .context("a" * 40)
            .context("a" * 40),
            """You are a friendly assistant
aa...
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa""",
            [(TruncateCompactor, str)],
            id="truncate_all_before_drop",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=1_000,
                token_counter=CharEstimateCounter(),
                compactors=(DropFragCompactor(),),
            )
            .system("You are a friendly assistant")
            .history("a" * ((1_000 * 4) + 100))
            .context("ctx"),
            "You are a friendly assistant\nctx",
            [(DropFragCompactor, str)],
            id="drop_history_before_context",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=1_000,
                token_counter=CharEstimateCounter(),
                compactors=(DropFragCompactor(),),
            )
            .system("You are a friendly assistant")
            .history("low hist", importance=Importance.LOW)
            .history("crit hist", importance=Importance.CRITICAL)
            .context("a" * ((1_000 * 4) + 100)),
            "You are a friendly assistant\ncrit hist",
            [(DropFragCompactor, str), (DropFragCompactor, str)],
            id="drop_lowest_importance_history_first",
        ),
        pytest.param(
            PromptBuilder(
                max_tokens=6,
                token_counter=CharEstimateCounter(),
                compactors=(InlineCompactor(),),
            )
            .system("You are a friendly assistant")
            .context(TestStruct()),
            "You are a friendly assistant\nThisisaverylongsentence",
            [(InlineCompactor, TestStruct)],
            id="simple_inline_compactor",
        ),
    ],
)
def test_compaction(
    pb: PromptBuilder,
    expected_prompt: str,
    expected_compactions: list[tuple[type[Compactor], type[Promptable] | str]],
):
    result = pb.build()
    compactions = [
        (event.compactor_name, event.fragment_type)
        for event in result.compaction_events
    ]
    expected_compactions_stringified = [
        tuple(t.__name__ for t in pair) for pair in expected_compactions
    ]
    assert result.prompt == expected_prompt
    assert compactions == expected_compactions_stringified


def test_custom_importance_overrides_default():
    pb = PromptBuilder(max_tokens=100_000, token_counter=CharEstimateCounter())
    pb.history("important history", importance=Importance.HIGH)
    assert pb._slots["history"][0].importance == Importance.HIGH


def test_char_estimate_over_budget():
    pb = (
        PromptBuilder(
            max_tokens=1_000,
            token_counter=CharEstimateCounter(),
            disable_compaction=True,
        )
        .system("You are a friendly assistant")
        .context("a" * ((1_000 * 4) + 100))
    )

    with pytest.raises(ValueError):
        pb.build()


def test_disable_compaction_within_budget_does_not_raise():
    pb = PromptBuilder(
        max_tokens=1_000, token_counter=CharEstimateCounter(), disable_compaction=True
    ).user("short")
    assert pb.build().prompt == "short"


def test_build_idempotency():
    pb = (
        PromptBuilder(
            max_tokens=1_000,
            token_counter=CharEstimateCounter(),
        )
        .system("You are a friendly assistant")
        .context("a" * ((1_000 * 4) + 100))
    )
    assert pb.build() == pb.build()


def test_exhaustion_policy_warning(caplog):
    pb = (
        PromptBuilder(max_tokens=5, token_counter=CharEstimateCounter())
        .system("You are a friendly assistant")
        .context(TestStruct())
    )

    with caplog.at_level(logging.WARNING, logger="headroom"):
        pb.build()

    assert "Prompt is still over budget" in caplog.text


def test_exhaustion_policy_raise():
    pb = (
        PromptBuilder(
            max_tokens=5,
            token_counter=CharEstimateCounter(),
            exhaustion_policy=ExhaustionPolicy.RAISE,
        )
        .system("You are a friendly assistant")
        .context(TestStruct())
    )

    with pytest.raises(ValueError) as excinfo:
        pb.build()

    assert "Prompt is still over budget" in str(excinfo.value)


def test_max_tokens_safety_hatch():
    pb = PromptBuilder(max_tokens=10, token_counter=CharEstimateCounter()).system(
        "You are a friendly assistant"
    )

    assert pb.build().token_budget == 10


def test_token_budget_follows_model_name():
    pb = PromptBuilder(model_name="gpt-4o").system("You are a friendly assistant")

    assert pb.build().token_budget, pb.get_encoder() == MODEL_REGISTRY["gpt-4o"]


@pytest.mark.parametrize(
    "model_name,expected_encoder",
    [
        ("gpt-4o", "o200k_base"),
    ],
)
def test_encoding(model_name: str | None, expected_encoder: str):
    pb = PromptBuilder(model_name=model_name).system("You are a friendly assistant.")

    assert pb.get_encoder() == expected_encoder


def test_max_tokens_overrides_model_context_window():
    pb = PromptBuilder(model_name="gpt-4o", max_tokens=10).system(
        "You are a friendly assistant."
    )
    assert pb.build().token_budget == 10


def test_builder_raises_when_no_model_or_max_tokens_provided():
    with pytest.raises(ValueError):
        PromptBuilder().system("Hello")


def test_builder_raises_for_unknown_model():
    with pytest.raises(ValueError):
        PromptBuilder(model_name="unknown").system("Hello")
