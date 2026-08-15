from urdu_interp.corpora import AlignedTriple
from urdu_interp.tasks import next_word_item, next_word_items


def test_next_word_item_splits_off_last_word():
    item = next_word_item("main tum se bohat mohabbat", "t1", "ur")
    assert item.prefix == "main tum se bohat"
    assert item.gold_word == "mohabbat"
    assert item.triple_id == "t1"
    assert item.lang == "ur"


def test_next_word_item_none_for_single_word():
    assert next_word_item("mohabbat", "t1", "ur") is None


def test_next_word_items_skips_missing_or_short_fields():
    triples = [
        AlignedTriple(id="a", source="s", ur="one two three", hi="ek do teen"),
        AlignedTriple(id="b", source="s", ur="single", hi="akela"),
        AlignedTriple(id="c", source="s", ur="", hi="khaali"),
    ]
    items = next_word_items(triples, lang="ur")
    assert len(items) == 1
    assert items[0].triple_id == "a"
