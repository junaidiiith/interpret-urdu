"""Word-level alignment built on top of corpora.AlignedTriple.

Only one alignment strategy is used: exact positional matching, valid when
two fields of a triple are transliterations of each other (same words, same
order -- true for the Rekhta ur/hi/en fields). No statistical word aligner is
used anywhere in this library; sentence/verse-level triples that are
independently authored translations (FLORES, eBible, OPUS pivot) are compared
as whole units in convergence.py instead of at the word level, because there
is no reliable free alignment finer than "these two sentences mean the same
thing".
"""
from __future__ import annotations

from dataclasses import dataclass

from urdu_interp.corpora import AlignedTriple


@dataclass(frozen=True)
class AlignedWord:
    triple_id: str
    index: int
    word_a: str
    word_b: str


@dataclass(frozen=True)
class AlignmentStats:
    total_triples: int
    aligned_triples: int
    dropped_triples: int
    total_word_pairs: int


def positional_word_pairs(text_a: str, text_b: str) -> list[tuple[str, str]] | None:
    """Pair up words by position; None if the two texts don't have the same
    word count (transliteration should preserve word count -- a mismatch means
    something upstream doesn't actually satisfy that assumption for this pair,
    so it's dropped rather than guessed at)."""
    words_a = text_a.split()
    words_b = text_b.split()
    if len(words_a) != len(words_b) or not words_a:
        return None
    return list(zip(words_a, words_b))


def align_words_positional(
    triples: list[AlignedTriple], lang_a: str = "ur", lang_b: str = "hi"
) -> tuple[list[AlignedWord], AlignmentStats]:
    aligned: list[AlignedWord] = []
    dropped = 0
    for triple in triples:
        text_a = getattr(triple, lang_a)
        text_b = getattr(triple, lang_b)
        if text_a is None or text_b is None:
            dropped += 1
            continue
        pairs = positional_word_pairs(text_a, text_b)
        if pairs is None:
            dropped += 1
            continue
        aligned.extend(
            AlignedWord(triple_id=triple.id, index=i, word_a=wa, word_b=wb)
            for i, (wa, wb) in enumerate(pairs)
        )
    stats = AlignmentStats(
        total_triples=len(triples),
        aligned_triples=len(triples) - dropped,
        dropped_triples=dropped,
        total_word_pairs=len(aligned),
    )
    return aligned, stats
