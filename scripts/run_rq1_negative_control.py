"""RQ1-4: negative control. RQ1's shuffled-pair null (built into
layerwise_convergence) checks whether *this* pair of languages converges
more than chance -- but nothing previously checked whether the pipeline
itself manufactures spurious "convergence" between two things that
shouldn't converge at all. This runs the identical pipeline on real Urdu
vectors against random vectors of matched shape/per-layer statistics; an
effect size near zero across all layers is what confirms the instrument
(not just the object of study) is behaving.

    python scripts/run_rq1_negative_control.py --model qwen3.5-0.8b-base \\
        --corpus rekhta_ghazals --limit 300
"""
from __future__ import annotations

import argparse

import torch

from urdu_interp import convergence, corpora, models, paths, pipeline

CORPUS_KIND = {"rekhta_ghazals": "ghazals", "rekhta_nazms": "nazms"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--corpus", default="rekhta_ghazals", choices=list(CORPUS_KIND))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    triples = corpora.load_rekhta(CORPUS_KIND[args.corpus], limit=args.limit)

    # lang_b="hi" only to get real word-level alignment/pooling; the Hindi
    # vectors themselves are discarded -- only the real Urdu vectors matter.
    batch = pipeline.extract_triple_word_vectors(model, tokenizer, triples, "ur", "hi", args.device)
    vecs_a = batch.vecs_a  # (N, num_layers, hidden_dim), real contextual Urdu vectors

    generator = torch.Generator().manual_seed(args.seed)
    per_layer_std = vecs_a.std(dim=0, keepdim=True)  # (1, num_layers, hidden_dim)
    random_vecs = torch.randn(vecs_a.shape, generator=generator) * per_layer_std

    result = convergence.layerwise_convergence(vecs_a, random_vecs, n_null=30, seed=args.seed)
    mean_effect = sum(result.effect_size) / len(result.effect_size)
    print(f"negative control ({args.model}/{args.corpus}, n={result.n_pairs} word pairs): "
          f"max effect_size={max(result.effect_size):.3f}, mean effect_size={mean_effect:.3f}")
    print("effect sizes should sit near 0 across every layer -- a real effect here would mean "
          "the pipeline manufactures spurious convergence on its own, independent of the languages compared.")
    for layer, es in enumerate(result.effect_size):
        print(f"  layer {layer}: effect_size={es:.3f} layer_score={result.layer_scores[layer]:.3f}")


if __name__ == "__main__":
    main()
