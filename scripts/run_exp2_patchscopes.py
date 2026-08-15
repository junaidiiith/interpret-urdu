"""RQ2 follow-up EXP-2: Patchscopes read-out, using the same eBible ur
next-word items the original RQ2 detour measurement uses, so this is
directly comparable to the raw/tuned lens's crossover-layer numbers on the
same source data -- just read out by the model's own generation instead of
a linear probe (see patchscopes.py).

Scope: Qwen3.5 and SmolLM3 only (causal.py's hook mechanism, which this
depends on, excludes Gemma-3n's AltUp architecture).

    python scripts/run_exp2_patchscopes.py --model qwen3.5-0.8b-base --device cuda:2
"""
from __future__ import annotations

import argparse

from urdu_interp import corpora, extraction, models, paths, patchscopes, reporting, script_utils, tasks
from urdu_interp.stats import first_stable_index

N_LAYER_POINTS = 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=25, help="number of eBible ur next-word items")
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    roman_urdu_words = corpora.load_roman_urdu_wordlist()

    triples = corpora.load_ebible(limit=args.limit)
    items = tasks.next_word_items(triples, lang="ur")
    print(f"[{args.model}] {len(items)} eBible ur next-word items")

    probe_hs = extraction.extract_hidden_states(model, tokenizer, items[0].prefix, device=args.device)
    num_layers = probe_hs.layers.shape[0]
    layers_to_test = sorted({max(1, round(i * (num_layers - 1) / (N_LAYER_POINTS - 1))) for i in range(N_LAYER_POINTS)})
    print(f"[{args.model}] num_layers={num_layers}, testing layers={layers_to_test}")

    rows = []
    per_layer_scripts: dict[int, list[str]] = {l: [] for l in layers_to_test}
    for item in items:
        for layer in layers_to_test:
            text = patchscopes.patch_and_verbalize(model, tokenizer, item.prefix, layer, args.device)
            first_word = text.split()[0] if text.split() else ""
            script = script_utils.classify_token(first_word) if first_word else "other"
            per_layer_scripts[layer].append(script)
            rows.append({"model": args.model, "triple_id": item.triple_id, "layer": layer, "num_layers": num_layers, "verbalization": text, "first_word": first_word, "script": script})

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RESULTS_DIR / "exp2_patchscopes" / f"{args.model}.csv"
    reporting.save_table(df, out_path)

    print(f"\n[{args.model}] script distribution of Patchscope verbalizations by patched layer:")
    layer_flags = []
    for layer in layers_to_test:
        scripts = per_layer_scripts[layer]
        n = len(scripts)
        frac_urdu = scripts.count("urdu") / n
        frac_hindi = scripts.count("hindi") / n
        frac_latin = scripts.count("latin") / n
        frac_other = scripts.count("other") / n
        print(f"  layer {layer}/{num_layers}: urdu={frac_urdu:.2f} hindi={frac_hindi:.2f} latin={frac_latin:.2f} other={frac_other:.2f}")
        layer_flags.append(frac_urdu > frac_hindi and frac_urdu > frac_latin)
    crossover = first_stable_index(layer_flags)
    crossover_layer = layers_to_test[crossover] if crossover is not None else None
    print(f"[{args.model}] earliest tested layer from which urdu-verbalization fraction stably leads: {crossover_layer}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
