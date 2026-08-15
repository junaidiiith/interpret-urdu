from urdu_interp.alignment import align_words_positional, positional_word_pairs
from urdu_interp.corpora import AlignedTriple


def test_positional_word_pairs_matches_by_index():
    pairs = positional_word_pairs("a b c", "x y z")
    assert pairs == [("a", "x"), ("b", "y"), ("c", "z")]


def test_positional_word_pairs_none_on_length_mismatch():
    assert positional_word_pairs("a b c", "x y") is None


def test_positional_word_pairs_none_on_empty():
    assert positional_word_pairs("", "") is None


def test_positional_word_pairs_handles_multiline():
    ur = "محبت تو مت کر\nدل اس سے"
    hi = "मोहब्बत तू मत कर\nदिल उस से"
    pairs = positional_word_pairs(ur, hi)
    assert pairs is not None
    assert len(pairs) == len(ur.split())


def test_positional_word_pairs_none_when_transliteration_splits_a_compound_differently():
    # a real failure mode in the Rekhta data: "بے وفا" (two Urdu words) is
    # sometimes transliterated as the single compound "बेवफ़ा" in Hindi --
    # the word counts genuinely don't match, so this must be dropped, not
    # force-aligned.
    ur = "دل اس بے وفا سے"
    hi = "दिल उस बेवफ़ा से"
    assert positional_word_pairs(ur, hi) is None


def test_align_words_positional_drops_mismatched_triples_and_counts_stats():
    triples = [
        AlignedTriple(id="a", source="s", ur="a b c", hi="x y z"),
        AlignedTriple(id="b", source="s", ur="a b", hi="x y z"),  # mismatched length
        AlignedTriple(id="c", source="s", ur="p q", hi="r s"),
    ]
    aligned, stats = align_words_positional(triples, lang_a="ur", lang_b="hi")

    assert stats.total_triples == 3
    assert stats.dropped_triples == 1
    assert stats.aligned_triples == 2
    assert stats.total_word_pairs == 5
    assert all(w.triple_id in {"a", "c"} for w in aligned)


def test_align_words_positional_drops_when_field_missing():
    triples = [AlignedTriple(id="a", source="s", ur="a b", hi=None)]
    aligned, stats = align_words_positional(triples, lang_a="ur", lang_b="hi")
    assert stats.dropped_triples == 1
    assert aligned == []
