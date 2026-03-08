from headroom.builder import PromptBuilder


class TestPromptBuilder:
    def test_raw_string(self):
        pb = PromptBuilder().context("hello")
        assert "hello" == pb.build()