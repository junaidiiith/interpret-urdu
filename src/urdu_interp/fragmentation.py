"""Tokenizer fragmentation: how many subword pieces a whitespace word gets
split into. Used as a covariate in RQ1 to check whether an early-layer
convergence dip is explained by script fragmentation rather than a deeper
representational gap."""
from __future__ import annotations


def fragmentation_ratio(tokenizer, text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    total_pieces = sum(len(tokenizer.tokenize(word)) for word in words)
    return total_pieces / len(words)


def per_word_fragmentation(tokenizer, text: str) -> list[int]:
    return [len(tokenizer.tokenize(word)) for word in text.split()]
