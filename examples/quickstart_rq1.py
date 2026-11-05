#!/usr/bin/env python3
"""Minimal RQ1 quickstart: Urdu--Hindi representational convergence on Rekhta.

Matches the paper's RQ1 measurement (centered cosine vs. shuffled-pair null)
on a small slice so newcomers can verify the install end-to-end.

    python examples/quickstart_rq1.py --device cuda:0 --limit 50
    python examples/quickstart_rq1.py --device cpu --limit 10   # slow, but works
"""
from __future__ import annotations

import argparse

from urdu_interp import convergence, corpora, models, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5-0.8b-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=50, help="max Rekhta triples")
    parser.add_argument("--n-null", type=int, default=30)
    args = parser.parse_args()

    spec = models.get_spec(args.model)
    print(f"loading {spec.key} ({spec.hf_repo}) on {args.device}")
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)

    triples = corpora.load_rekhta(kind="ghazals", limit=args.limit)
    print(f"loaded {len(triples)} Rekhta ghazal triples")

    batch = pipeline.extract_triple_word_vectors(
        model, tokenizer, triples, lang_a="ur", lang_b="hi", device=args.device
    )
    print(f"aligned word pairs: {batch.stats}")

    result = convergence.layerwise_convergence(
        batch.vecs_a, batch.vecs_b, n_null=args.n_null
    )
    auc = convergence.effect_size_auc(result.effect_size)
    peak = convergence.peak_layer_frac(result.layer_scores)
    print(f"effect_size_auc = {auc:.4f}")
    print(f"peak_layer_frac = {peak:.4f}")
    print("per-layer effect sizes (first 5):", [round(x, 3) for x in result.effect_size[:5]])


if __name__ == "__main__":
    main()
