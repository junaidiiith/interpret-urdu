"""RQ2 follow-up EXP-8: TCAV-style concept-attribution significance test.
Fits a real linear-classifier CAV for "English vs. Urdu" at a layer, tests
its causal effect on real output logits (same statistic family as EXP-4),
and compares that effect against a null distribution of CAVs fit on
randomly permuted labels over the same activations -- the negative control
EXP-4/EXP-5 never had. See concept_attribution.py for the full method note.

    python scripts/run_exp8_concept_significance.py --model qwen3.5-0.8b-base --device cuda:2
"""
from __future__ import annotations

import argparse
import json

from urdu_interp import concept_attribution, corpora, extraction, models, paths, script_utils, tasks

# (layer_frac, label) pairs -- deliberately spans EXP-4's "weak effect" and
# "dramatic effect" depths so the significance test's verdict can be
# compared at both ends, not just the most dramatic layer.
LAYER_FRACS = [("weak", 0.48), ("dramatic", 0.88)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-fit-pairs", type=int, default=80, help="FLORES ur/en sentence pairs used to fit the CAV")
    parser.add_argument("--n-items", type=int, default=60, help="eBible ur next-word items used to measure causal effect")
    parser.add_argument("--n-null", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    roman_urdu_words = corpora.load_roman_urdu_wordlist()
    latin_subtype_fn = lambda t: script_utils.classify_latin_subtype(t, roman_urdu_words)  # noqa: E731

    fit_triples = corpora.load_flores("dev", limit=args.n_fit_pairs)
    ur_fit_texts = [t.ur for t in fit_triples if t.ur]
    en_fit_texts = [t.en for t in fit_triples if t.en]

    eval_triples = corpora.load_ebible(limit=args.n_items)
    prefixes = [i.prefix for i in tasks.next_word_items(eval_triples, lang="ur") if i.prefix][: args.n_items]
    print(f"[{args.model}] {len(ur_fit_texts)} ur / {len(en_fit_texts)} en fit sentences, {len(prefixes)} eval prefixes")

    probe_hs = extraction.extract_hidden_states(model, tokenizer, prefixes[0], device=args.device)
    num_layers = probe_hs.layers.shape[0]

    report = {"model": args.model, "n_null": args.n_null, "alpha": args.alpha, "layers": {}}
    for label, frac in LAYER_FRACS:
        layer = max(1, round(frac * (num_layers - 1)))
        print(f"\n[{args.model}] === layer {layer}/{num_layers} ({label}, {frac:.0%} depth) ===")

        en_vecs = concept_attribution.sentence_vectors_at_layer(model, tokenizer, en_fit_texts, layer, args.device)
        ur_vecs = concept_attribution.sentence_vectors_at_layer(model, tokenizer, ur_fit_texts, layer, args.device)

        real_cav = concept_attribution.fit_cav(en_vecs, ur_vecs, seed=0)
        null_cavs = concept_attribution.fit_null_cavs(en_vecs, ur_vecs, n_null=args.n_null, seed=0)

        real_result = concept_attribution.measure_cav_effect(
            model, tokenizer, real_cav, prefixes, layer, args.alpha, args.device, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn
        )
        print(f"[{args.model}] real CAV: mean_delta_latin={real_result.mean_delta_latin:+.4f}, mean_delta_urdu={real_result.mean_delta_urdu:+.4f}")

        null_effects = []
        for i, null_cav in enumerate(null_cavs):
            null_result = concept_attribution.measure_cav_effect(
                model, tokenizer, null_cav, prefixes, layer, args.alpha, args.device, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn
            )
            null_effects.append(null_result.mean_delta_latin)
            print(f"[{args.model}]   null CAV {i + 1}/{args.n_null}: mean_delta_latin={null_result.mean_delta_latin:+.4f}")

        sig = concept_attribution.permutation_test(real_result.mean_delta_latin, null_effects)
        print(f"[{args.model}] {label} layer significance: real={sig['real_effect']:+.4f}, "
              f"null mean/std={sig['null_mean']:+.4f}/{sig['null_std']:.4f}, z={sig['z_score']:.2f}, p={sig['p_value']:.4f}")

        report["layers"][label] = {
            "layer": layer,
            "num_layers": num_layers,
            "real_mean_delta_urdu": real_result.mean_delta_urdu,
            "real_mean_delta_latin": real_result.mean_delta_latin,
            "real_mean_delta_hindi": real_result.mean_delta_hindi,
            "null_effects_delta_latin": null_effects,
            "significance": sig,
        }

    out_path = paths.RESULTS_DIR / "exp8_concept_significance" / f"{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
