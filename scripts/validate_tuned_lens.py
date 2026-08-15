"""RQ2 follow-up EXP-1: is the tuned lens (RQ2-1) actually calibrated, or was
its extreme Qwen layer-0 crossover result an artifact of an unvalidated,
underdetermined fit? Fits at several sample sizes, evaluates each fit on a
disjoint held-out set (never seen during fitting), and reports held-out R^2 /
top-1 agreement per layer plus the crossover-layer readout at each fit size.

    python scripts/validate_tuned_lens.py --model qwen3.5-0.8b-base --device cuda:2
"""
from __future__ import annotations

import argparse
import json

from urdu_interp import corpora, models, paths, pipeline, script_utils, tasks, tuned_lens
from urdu_interp.logit_lens import native_crossover_layer

FIT_SIZES = [100, 300, 600]


def crossover_at(model, tokenizer, lens_params, device: str, n_items: int = 100) -> int | None:
    unembed_fns = tuned_lens.make_tuned_unembed_fns(model, lens_params)
    roman_urdu_words = corpora.load_roman_urdu_wordlist()
    latin_subtype_fn = lambda t: script_utils.classify_latin_subtype(t, roman_urdu_words)  # noqa: E731
    triples = corpora.load_ebible(limit=n_items)
    items = tasks.next_word_items(triples, lang="ur")
    results = pipeline.run_detour_for_items(
        model, tokenizer, items, unembed_fns, "urdu", ["hindi", "latin"], device, latin_subtype_fn=latin_subtype_fn
    )
    layers = [d.native_crossover_layer for _, d in results if d.native_crossover_layer is not None]
    if not layers:
        return None
    return sorted(layers)[len(layers) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--holdout-size", type=int, default=200)
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)

    # fit and holdout pools drawn from disjoint corpora entirely (fit: flores
    # dev, holdout: ebible) so there's no accidental sentence overlap even
    # though both are "ordinary formal text" -- a stricter separation than
    # run_rq2_logit_lens.py's default (which doesn't need one, since it never
    # evaluates the lens against itself).
    holdout_triples = corpora.load_ebible(limit=args.holdout_size)
    holdout_texts = [t.ur for t in holdout_triples if t.ur][: args.holdout_size]

    report = {"model": args.model, "fit_sizes": {}}
    for fit_size in FIT_SIZES:
        fit_triples = corpora.load_flores("dev", limit=fit_size)
        fit_texts = [t.ur for t in fit_triples if t.ur] + [t.hi for t in fit_triples if t.hi] + [t.en for t in fit_triples if t.en]
        print(f"[{args.model}] fitting tuned lens on {len(fit_texts)} texts (target fit_size={fit_size})...")
        lens_params = tuned_lens.fit_tuned_lens(model, tokenizer, fit_texts, args.device)

        print(f"[{args.model}] evaluating on {len(holdout_texts)} held-out texts...")
        eval_rows = tuned_lens.evaluate_tuned_lens(model, tokenizer, lens_params, holdout_texts, args.device)

        print(f"[{args.model}] computing crossover layer at this fit size...")
        crossover = crossover_at(model, tokenizer, lens_params, args.device)

        num_layers = len(eval_rows)
        early = eval_rows[: num_layers // 3]
        late = eval_rows[-num_layers // 3 :]
        summary = {
            "n_fit_texts": len(fit_texts),
            "crossover_layer": crossover,
            "num_layers": num_layers,
            "mean_r2_holdout_early_third": sum(r["r2_holdout"] for r in early) / len(early),
            "mean_r2_holdout_late_third": sum(r["r2_holdout"] for r in late) / len(late),
            "mean_top1_agreement_early_third": sum(r["top1_agreement"] for r in early) / len(early),
            "mean_top1_agreement_late_third": sum(r["top1_agreement"] for r in late) / len(late),
            "top1_agreement_final_layer": eval_rows[-1]["top1_agreement"],
            "per_layer": eval_rows,
        }
        report["fit_sizes"][fit_size] = summary
        print(f"[{args.model}] fit_size={fit_size}: crossover={crossover}, "
              f"R2(early/late thirds)={summary['mean_r2_holdout_early_third']:.3f}/{summary['mean_r2_holdout_late_third']:.3f}, "
              f"top1_agree(early/late thirds)={summary['mean_top1_agreement_early_third']:.3f}/{summary['mean_top1_agreement_late_third']:.3f}, "
              f"top1_agree(final layer)={summary['top1_agreement_final_layer']:.3f}")

    out_path = paths.RESULTS_DIR / "tuned_lens_validation" / f"{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
