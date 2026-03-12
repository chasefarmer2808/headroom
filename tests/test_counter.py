from headroom.counter import CharEstimateCounter


def test_char_estimate_known_str():
    assert 4 == CharEstimateCounter().count_tokens("how are you today?")
