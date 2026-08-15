"""RQ2 follow-up EXP-4: causal test of the "detour." Everything in RQ2 so
far only reads an intermediate representation through some lens; this patches
a hidden state captured from the *English* translation of a sentence into the
model's forward pass on the *Urdu* version of the same sentence (content-
matched via existing FLORES triples, no new data), at the same layer and the
last token position, then reads the model's own REAL final-layer logits (not
a lens) to see whether the swap causally shifts the actual next-token
distribution -- not just an intermediate readout.

Scope: Qwen3.5 and SmolLM3 only (see causal.py's docstring for why Gemma-3n's
AltUp architecture is excluded from this module).

    python scripts/run_exp4_causal_patching.py --model qwen3.5-0.8b-base --device cuda:2
"""
from __future__ import annotations

import argparse

from urdu_interp import causal, corpora, extraction, logit_lens, models, paths, reporting, script_utils, tasks

LAYER_FRACS = [0.25, 0.5, 0.75, 0.9]


def _decode_id_fn(tokenizer):
    def decode_id(tid: int) -> str:
        return tokenizer.decode([tid])

    return decode_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=40, help="number of FLORES triples to test")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    decode_id = _decode_id_fn(tokenizer)
    roman_urdu_words = corpora.load_roman_urdu_wordlist()
    latin_subtype_fn = lambda t: script_utils.classify_latin_subtype(t, roman_urdu_words)  # noqa: E731

    triples = corpora.load_flores("dev", limit=args.limit)
    ur_items = {i.triple_id: i for i in tasks.next_word_items(triples, lang="ur")}
    en_items = {i.triple_id: i for i in tasks.next_word_items(triples, lang="en")}
    shared_ids = [tid for tid in ur_items if tid in en_items]
    print(f"[{args.model}] {len(shared_ids)} content-matched ur/en prefix pairs")

    # probe once to get num_layers
    probe_hs = extraction.extract_hidden_states(model, tokenizer, ur_items[shared_ids[0]].prefix, device=args.device)
    num_layers = probe_hs.layers.shape[0]
    layers_to_test = sorted({max(1, round(f * (num_layers - 1))) for f in LAYER_FRACS})
    print(f"[{args.model}] num_layers={num_layers}, testing layers={layers_to_test}")

    rows = []
    for tid in shared_ids:
        ur_prefix = ur_items[tid].prefix
        en_prefix = en_items[tid].prefix
        if not ur_prefix or not en_prefix:
            continue

        hs_en = extraction.extract_hidden_states(model, tokenizer, en_prefix, device=args.device)
        enc_ur = tokenizer(ur_prefix, return_tensors="pt").to(args.device)

        import torch

        with torch.no_grad():
            baseline_logits = model(**enc_ur).logits[0, -1, :].float().cpu()
        baseline_mass = logit_lens.layer_script_mass(baseline_logits, decode_id, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn)

        for layer in layers_to_test:
            if layer >= hs_en.layers.shape[0]:
                continue
            replacement = hs_en.layers[layer, -1, :]
            with causal.patch_hidden_state(model, layer=layer, position=-1, replacement=replacement):
                with torch.no_grad():
                    patched_logits = model(**enc_ur).logits[0, -1, :].float().cpu()
            patched_mass = logit_lens.layer_script_mass(patched_logits, decode_id, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn)

            rows.append(
                {
                    "model": args.model,
                    "triple_id": tid,
                    "patch_layer": layer,
                    "num_layers": num_layers,
                    "baseline_mass_urdu": baseline_mass["urdu"],
                    "patched_mass_urdu": patched_mass["urdu"],
                    "delta_mass_urdu": patched_mass["urdu"] - baseline_mass["urdu"],
                    "baseline_mass_latin": baseline_mass["latin"],
                    "patched_mass_latin": patched_mass["latin"],
                    "delta_mass_latin": patched_mass["latin"] - baseline_mass["latin"],
                    "baseline_mass_hindi": baseline_mass["hindi"],
                    "patched_mass_hindi": patched_mass["hindi"],
                    "delta_mass_hindi": patched_mass["hindi"] - baseline_mass["hindi"],
                }
            )

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RESULTS_DIR / "exp4_causal_patching" / f"{args.model}.csv"
    reporting.save_table(df, out_path)

    print(f"\n[{args.model}] mean delta_mass by patch layer (patched-english-state minus baseline):")
    for layer in layers_to_test:
        sub = df[df["patch_layer"] == layer]
        if len(sub) == 0:
            continue
        print(
            f"  layer {layer}/{num_layers}: n={len(sub)}, "
            f"delta_urdu={sub['delta_mass_urdu'].mean():+.4f}, "
            f"delta_latin={sub['delta_mass_latin'].mean():+.4f}, "
            f"delta_hindi={sub['delta_mass_hindi'].mean():+.4f}"
        )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
