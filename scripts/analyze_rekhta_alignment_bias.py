"""DAT-1: the Rekhta ur-hi word-alignment sample is a small survivor pool
(exact word-count matching keeps roughly 6-7% of triples) -- and survivorship
is not obviously random (compound/izafat-heavy lines are the systematic
failure mode, per EXECUTION_PLAN.md's own note about Devanagari transliterating
Persian izafat constructions as hyphenated single "words"). This compares
word count, line count, and a type-token-ratio lexical-density proxy between
the triples that survive alignment and a random sample of the ones that get
dropped, so the bias is reported directly instead of left implicit.

    python scripts/analyze_rekhta_alignment_bias.py --corpus rekhta_ghazals
"""
from __future__ import annotations

import argparse
import random
import statistics

from urdu_interp import alignment, corpora


def _stats(triples: list[corpora.AlignedTriple]) -> dict[str, float]:
    word_counts = [len(t.ur.split()) for t in triples]
    line_counts = [t.ur.count("\n") + 1 for t in triples]
    ttrs = []
    for t in triples:
        words = t.ur.split()
        if words:
            ttrs.append(len(set(words)) / len(words))
    return {
        "n": len(triples),
        "mean_word_count": statistics.mean(word_counts) if word_counts else 0.0,
        "median_word_count": statistics.median(word_counts) if word_counts else 0.0,
        "mean_line_count": statistics.mean(line_counts) if line_counts else 0.0,
        "mean_type_token_ratio": statistics.mean(ttrs) if ttrs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="rekhta_ghazals", choices=["rekhta_ghazals", "rekhta_nazms"])
    parser.add_argument("--sample-size", type=int, default=None, help="random sample size from the dropped pool (default: same size as the surviving pool)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kind = "ghazals" if args.corpus == "rekhta_ghazals" else "nazms"
    triples = corpora.load_rekhta(kind)

    survivors = []
    dropped = []
    for t in triples:
        pairs = alignment.positional_word_pairs(t.ur, t.hi) if t.hi else None
        (survivors if pairs is not None else dropped).append(t)

    print(f"{args.corpus}: {len(triples)} total triples, {len(survivors)} survive ur-hi word-count alignment "
          f"({100 * len(survivors) / len(triples):.1f}%), {len(dropped)} dropped")

    rng = random.Random(args.seed)
    sample_size = args.sample_size or len(survivors)
    dropped_sample = rng.sample(dropped, min(sample_size, len(dropped)))

    survivor_stats = _stats(survivors)
    dropped_stats = _stats(dropped_sample)

    print(f"\n{'metric':<28}{'survivors':>15}{'dropped (sample)':>20}{'ratio (dropped/survivors)':>28}")
    for key in ("n", "mean_word_count", "median_word_count", "mean_line_count", "mean_type_token_ratio"):
        s, d = survivor_stats[key], dropped_stats[key]
        ratio = f"{d / s:.2f}x" if s else "n/a"
        print(f"{key:<28}{s:>15.2f}{d:>20.2f}{ratio:>28}")

    print("\nInterpretation: if dropped triples run substantially longer (more words/lines) than survivors, "
          "the survivor pool is biased toward shorter, simpler lines -- exactly the register (compound/izafat-"
          "heavy, longer constructions) EXECUTION_PLAN.md already flagged as the systematic failure mode for "
          "exact ur/hi word-count matching, not random noise.")


if __name__ == "__main__":
    main()
