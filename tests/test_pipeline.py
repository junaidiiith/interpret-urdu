from urdu_interp.corpora import AlignedTriple
from urdu_interp.pipeline import fragmentation_covariate


class FakeTokenizer:
    def tokenize(self, word: str) -> list[str]:
        return list(word)  # one piece per character


def test_fragmentation_covariate_skips_missing_language_field():
    triples = [
        AlignedTriple(id="a", source="s", ur="ab cd", hi="x"),
        AlignedTriple(id="b", source="s", ur="", hi="y"),
    ]
    ratios = fragmentation_covariate(FakeTokenizer(), triples, lang="ur")
    assert len(ratios) == 1
    assert ratios[0] == (2 + 2) / 2  # "ab" and "cd" each split into 2 chars


def test_fragmentation_covariate_respects_max_triples():
    triples = [AlignedTriple(id=str(i), source="s", ur="ab", hi="x") for i in range(5)]
    ratios = fragmentation_covariate(FakeTokenizer(), triples, lang="ur", max_triples=2)
    assert len(ratios) == 2
