"""RQ2 follow-up EXP-3: word-repetition concept probes with a known correct
token per language, replacing RQ2's noisy script-bucket-over-free-text
measurement with a per-item ground truth (see concept_probes.py for the
task design and its caveats).

    python scripts/run_exp3_concept_probes.py --model qwen3.5-0.8b-base --device cuda:2
"""
from __future__ import annotations

import argparse
import json

from urdu_interp import concept_probes, extraction, models, paths, reporting
from urdu_interp.logit_lens import make_unembed_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concepts-file", default=str(paths.DATA_DIR / "qa" / "concept_probes.json"))
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    unembed_fn = make_unembed_fn(model)

    with open(args.concepts_file, encoding="utf-8") as f:
        concepts = json.load(f)

    rows = []
    crossover_layers = []
    n_resolved_correctly = 0
    for c in concepts:
        ur_id = concept_probes.first_token_id(tokenizer, c["ur"])
        hi_id = concept_probes.first_token_id(tokenizer, c["hi"])
        en_id = concept_probes.first_token_id(tokenizer, c["en"])
        # if two of the three languages collide on the same first token
        # (can happen with short/ambiguous tokenizations), the probe can't
        # distinguish them for this item -- skip rather than silently score
        # a degenerate three-way race.
        if len({ur_id, hi_id, en_id}) < 3:
            print(f"[{args.model}] skipping {c['concept']!r}: token collision (ur={ur_id}, hi={hi_id}, en={en_id})")
            continue

        prompt = concept_probes.build_prompt(c["ur"])
        hs = extraction.extract_hidden_states(model, tokenizer, prompt, device=args.device)
        layer_probs = concept_probes.run_concept_probe(hs, unembed_fn, ur_id, hi_id, en_id)
        crossover = concept_probes.concept_crossover_layer(layer_probs)
        resolved_correctly = layer_probs[-1]["ur"] > layer_probs[-1]["hi"] and layer_probs[-1]["ur"] > layer_probs[-1]["en"]
        n_resolved_correctly += int(resolved_correctly)
        if crossover is not None:
            crossover_layers.append(crossover)

        for layer, probs in enumerate(layer_probs):
            rows.append(
                {
                    "model": args.model,
                    "concept_id": c["id"],
                    "concept": c["concept"],
                    "layer": layer,
                    "p_ur": probs["ur"],
                    "p_hi": probs["hi"],
                    "p_en": probs["en"],
                    "crossover_layer": crossover,
                    "resolved_correctly_final_layer": resolved_correctly,
                }
            )

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RESULTS_DIR / "exp3_concept_probes" / f"{args.model}.csv"
    reporting.save_table(df, out_path)

    n_items = len(concepts)
    n_scored = n_items - (n_items - len({r["concept_id"] for r in rows}))
    n_scored = len({r["concept_id"] for r in rows})
    print(f"\n{args.model}: {n_resolved_correctly}/{n_scored} items resolve to the correct (Urdu, repeat-task) token by the final layer")
    if crossover_layers:
        median = sorted(crossover_layers)[len(crossover_layers) // 2]
        print(f"{args.model}: median crossover layer={median}, resolved for {len(crossover_layers)}/{n_scored} items")
    else:
        print(f"{args.model}: no item ever reached a stable Urdu-favoring crossover")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
