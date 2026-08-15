"""RQ3 step 2: rerun the RQ1 convergence measurement on a fine-tuned model and
its own pre-fine-tuning base, then report the paired per-layer delta -- does
fine-tuning move the layers RQ1 flagged as the deep gap, or only the early
ones?

    python scripts/run_rq3_eval.py --base qwen3.5-0.8b-base \\
        --finetuned-dir results/finetuned_models/qwen3.5-0.8b-base-urdu-ft \\
        --corpus rekhta_ghazals --lang-pair ur-hi --limit 200
"""
from __future__ import annotations

import argparse

import torch

from urdu_interp import convergence, corpora, models, paths, pipeline, reporting, stats
from urdu_interp.models import ModelSpec, load_model_and_tokenizer

CORPUS_LOADERS = {
    "rekhta_ghazals": lambda limit: corpora.load_rekhta("ghazals", limit=limit),
    "rekhta_nazms": lambda limit: corpora.load_rekhta("nazms", limit=limit),
}


def _local_spec(path: str) -> ModelSpec:
    return ModelSpec(key="finetuned", role="eval", hf_repo=path, revision=None, gated=False, variant="base", family="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base model key from configs/models.yaml")
    parser.add_argument("--finetuned-dir", required=True, help="local path to the fine-tuned checkpoint")
    parser.add_argument("--corpus", default="rekhta_ghazals", choices=list(CORPUS_LOADERS))
    parser.add_argument("--lang-pair", default="ur-hi")
    parser.add_argument("--limit", type=int, default=200, help="number of poems used for fine-tuning (must match run_rq3_finetune.py's --limit) -- NOT the eval set, see --eval-holdout")
    parser.add_argument("--eval-holdout", type=int, default=300, help="number of held-out poems to evaluate on (must match run_rq3_finetune.py's --eval-holdout)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-suffix", default="", help="appended to the output filename -- use for RQ3-4's hyperparameter-grid reruns so they don't overwrite the canonical fine-tuning arm's result")
    args = parser.parse_args()

    paths.ensure_dirs()
    lang_a, lang_b = args.lang_pair.split("-")
    # run_rq3_finetune.py trains on triples[:args.limit] and holds out
    # triples[args.limit:]; evaluating on the training slice would measure
    # memorization of the exact fine-tuning poems, not a general
    # representational shift -- load the same limit+holdout span and slice
    # to the held-out tail, exactly mirroring run_rq3_finetune.py.
    all_triples = CORPUS_LOADERS[args.corpus](args.limit + args.eval_holdout)
    triples = all_triples[args.limit:]
    print(f"evaluating on {len(triples)} held-out triples (poems {args.limit}:{args.limit + args.eval_holdout}, disjoint from fine-tuning data)")

    rows = []
    for stage, spec in (("before", models.get_spec(args.base)), ("after", _local_spec(args.finetuned_dir))):
        model, tokenizer = load_model_and_tokenizer(spec, device=args.device)
        batch = pipeline.extract_triple_word_vectors(model, tokenizer, triples, lang_a, lang_b, args.device)
        print(f"{stage}: {batch.stats}")
        result = convergence.layerwise_convergence(batch.vecs_a, batch.vecs_b)
        for row in reporting.convergence_rows(f"{args.base}[{stage}]", args.corpus, args.lang_pair, result):
            row["stage"] = stage
            rows.append(row)
        del model
        torch.cuda.empty_cache()

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RQ3_DIR / f"{args.base}__{args.corpus}__{args.lang_pair}_before_after{args.output_suffix}.csv"
    reporting.save_table(df, out_path)

    before = df[df["stage"] == "before"].sort_values("layer")["effect_size"].tolist()
    after = df[df["stage"] == "after"].sort_values("layer")["effect_size"].tolist()
    if len(before) == len(after) and len(before) > 1:
        wilcoxon_stat, wilcoxon_p = stats.paired_delta_test(before, after)
        # STA-1: adjacent layers' effect sizes are strongly autocorrelated,
        # so the per-layer Wilcoxon above is pseudoreplication -- keep it for
        # continuity but prefer the block-bootstrap p-value, which resamples
        # contiguous layer blocks instead of treating each layer as an
        # independent draw.
        boot_delta, boot_p = stats.block_bootstrap_paired_test(before, after)
        print(f"paired delta test (per-layer effect size, before vs after fine-tuning), "
              f"NOT autocorrelation-aware: wilcoxon stat={wilcoxon_stat:.3f} p={wilcoxon_p:.4f}")
        print(f"block-bootstrap paired test (autocorrelation-aware, preferred): "
              f"mean delta={boot_delta:.3f} p={boot_p:.4f}")

        # STA-4: fit the early/late split point instead of assuming the
        # network's exact midpoint is where fine-tuning's effect changes
        # character.
        delta_curve = [a - b for a, b in zip(after, before)]
        changepoint, sse = stats.fit_piecewise_changepoint(delta_curve)
        print(f"fitted changepoint (STA-4, vs. fixed midpoint {len(before) // 2}): "
              f"layer {changepoint} of {len(before)} (sse={sse:.4f})")
        early_fitted = delta_curve[:changepoint]
        late_fitted = delta_curve[changepoint:]
        print(f"fitted-split early-half mean delta: {sum(early_fitted) / len(early_fitted):.3f}" if early_fitted else "fitted-split early half empty")
        print(f"fitted-split late-half mean delta:  {sum(late_fitted) / len(late_fitted):.3f}" if late_fitted else "fitted-split late half empty")

        deep_half = len(before) // 2
        print(f"fixed-midpoint early-half mean delta: {sum(delta_curve[:deep_half]) / deep_half:.3f}")
        print(f"fixed-midpoint late-half mean delta:  {sum(delta_curve[deep_half:]) / (len(before) - deep_half):.3f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
