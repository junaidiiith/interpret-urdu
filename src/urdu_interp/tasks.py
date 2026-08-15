"""RQ2 task construction: turn an AlignedTriple into a next-word prediction
item, entirely automatically (the 'gold' word is just whatever word actually
came next in the source text -- no human authoring, no LLM involved)."""
from __future__ import annotations

from dataclasses import dataclass

from urdu_interp.corpora import AlignedTriple


@dataclass(frozen=True)
class NextWordItem:
    triple_id: str
    lang: str
    prefix: str
    gold_word: str


def next_word_item(text: str, triple_id: str, lang: str) -> NextWordItem | None:
    words = text.split()
    if len(words) < 2:
        return None
    return NextWordItem(triple_id=triple_id, lang=lang, prefix=" ".join(words[:-1]), gold_word=words[-1])


def next_word_items(triples: list[AlignedTriple], lang: str) -> list[NextWordItem]:
    items = []
    for t in triples:
        text = getattr(t, lang)
        if not text:
            continue
        item = next_word_item(text, t.id, lang)
        if item is not None:
            items.append(item)
    return items
