from urdu_interp.fragmentation import fragmentation_ratio, per_word_fragmentation


class FakeTokenizer:
    """Splits a word into one piece per 3 characters -- deterministic and
    independent of any real tokenizer, just enough to exercise the ratio math."""

    def tokenize(self, word: str) -> list[str]:
        return [word[i : i + 3] for i in range(0, len(word), 3)] or [word]


def test_fragmentation_ratio_averages_pieces_per_word():
    tok = FakeTokenizer()
    # "abcdef" -> 2 pieces, "xyz" -> 1 piece
    ratio = fragmentation_ratio(tok, "abcdef xyz")
    assert ratio == (2 + 1) / 2


def test_fragmentation_ratio_empty_text_is_zero():
    assert fragmentation_ratio(FakeTokenizer(), "") == 0.0


def test_per_word_fragmentation_returns_one_count_per_word():
    tok = FakeTokenizer()
    assert per_word_fragmentation(tok, "abcdef xyz a") == [2, 1, 1]
