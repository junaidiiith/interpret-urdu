"""ENG-3: report block count and a bootstrap confidence interval on held-out
perplexity alongside every before/after number -- eval_perplexity already
computed the per-block losses needed for this; it just discarded them and
returned only the mean. Eval-only against the already-saved fine-tuned
checkpoint (no retraining).

    python scripts/run_rq3_perplexity_ci.py --model qwen3.5-0.8b-base \\
        --finetuned-dir results/finetuned_models/qwen3.5-0.8b-base-urdu-ft \\
        --limit 3000 --eval-holdout 300 --device cuda:0
"""
from __future__ import annotations

import argparse
import json

import torch

from urdu_interp import corpora, finetune, models, paths
from urdu_interp.models import ModelSpec, load_model_and_tokenizer


def _local_spec(path: str) -> ModelSpec:
    return ModelSpec(key="finetuned", role="eval", hf_repo=path, revision=None, gated=False, variant="base", family="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base model key from configs/models.yaml")
    parser.add_argument("--finetuned-dir", required=True)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--eval-holdout", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths.ensure_dirs()
    all_triples = corpora.load_rekhta("ghazals", limit=args.limit + args.eval_holdout)
    holdout_triples = all_triples[args.limit :]
    holdout_texts = finetune.rekhta_training_texts(holdout_triples, lang="ur")

    results = {}
    for stage, spec in (("before", models.get_spec(args.model)), ("after", _local_spec(args.finetuned_dir))):
        model, tokenizer = load_model_and_tokenizer(spec, device=args.device)
        r = finetune.eval_perplexity_detailed(model, tokenizer, holdout_texts, block_size=args.block_size, device=args.device)
        results[stage] = r
        print(f"{stage}: perplexity={r.perplexity:.3f} (95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}], n_blocks={r.n_blocks})")
        del model
        torch.cuda.empty_cache()

    pct_change = 100 * (results["after"].perplexity - results["before"].perplexity) / results["before"].perplexity
    print(f"perplexity change: {pct_change:+.1f}%")
    ci_overlap = results["before"].ci_low <= results["after"].ci_high and results["after"].ci_low <= results["before"].ci_high
    print(f"before/after 95% CIs {'OVERLAP -- treat the drop with caution' if ci_overlap else 'do not overlap'}")

    out = {
        "model": args.model,
        "before": {"perplexity": results["before"].perplexity, "ci_low": results["before"].ci_low, "ci_high": results["before"].ci_high, "n_blocks": results["before"].n_blocks},
        "after": {"perplexity": results["after"].perplexity, "ci_low": results["after"].ci_low, "ci_high": results["after"].ci_high, "n_blocks": results["after"].n_blocks},
        "pct_change": pct_change,
    }
    out_path = paths.RQ3_DIR / f"{args.model}__perplexity_ci.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
