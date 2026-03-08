from headroom.builder import PromptBuilder


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