import pytest

from headroom.builder import DropSlotCompactor, PromptBuilder, TruncateCompactor


class TestPromptBuilder:
    def test_raw_string(self):
        pb = PromptBuilder().context("hello")
        assert "hello" == pb.build()

    def test_slot_ordering(self):
        pb = PromptBuilder() \
            .system("1") \
            .instructions("2") \
            .context("3") \
            .history("4") \
            .user("5")
        
        assert """1
2
3
4
5""" == pb.build()
        
    def test_char_estimate_over_budget(self):
        pb = PromptBuilder(disable_compaction=True).system("You are a friendly assistant").context("a" * ((1_000 * 4) + 100))

        with pytest.raises(ValueError):
            pb.build()

class TestCompaction:
    def test_drops_least_important_slot(self):
        pb = PromptBuilder().system("You are a friendly assistant").context("a" * ((1_000 * 4) + 100))

        assert "You are a friendly assistant" == pb.build()
    
    def test_exhaustive_compaction(self):
        large_context = "a" * ((1_000 * 4) + 100)
        pb = PromptBuilder().system("You are a friendly assistant").context(large_context).context(large_context)

        assert "You are a friendly assistant" == pb.build()


    def test_exhaustive_sequential_compaction(self):
        # pb with truncate compactor and drop frag compactor.
        # should truncate all of the frags before dropping any.
        pb = PromptBuilder(
            max_tokens=20,
            compactors=(TruncateCompactor(max_chars=5), DropSlotCompactor())
        ) \
        .system("You are a friendly assistant") \
        .context("a" * 40) \
        .context("a" * 40)

        assert """Yo...
aa...
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa""" == pb.build()
