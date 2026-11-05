#!/usr/bin/env python3
"""Minimal RQ2 causal quickstart: patch an English hidden state into an Urdu pass.

This mirrors the paper's causal test for a hidden English pathway
(`scripts/run_exp4_causal_patching.py`), on a tiny FLORES slice.

Scoped to Qwen3.5 / SmolLM3 (same as the paper).

    python examples/quickstart_rq2_causal.py --device cuda:0 --limit 8
"""
from __future__ import annotations

import argparse

import torch

from urdu_interp import causal, corpora, extraction, logit_lens, models, script_utils, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5-0.8b-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--layer-frac", type=float, default=0.88)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    if "gemma" in args.model:
        raise SystemExit("Causal patching excludes Gemma-3n (AltUp); use a Qwen3.5 or SmolLM3 key.")

    spec = models.get_spec(args.model)
    print(f"loading {spec.key} on {args.device}")
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)

    roman_urdu_words = corpora.load_roman_urdu_wordlist()
    latin_subtype_fn = lambda t: script_utils.classify_latin_subtype(t, roman_urdu_words)
    decode_id = lambda tid: tokenizer.decode([tid])

    triples = corpora.load_flores("dev", limit=args.limit)
    ur_items = {i.triple_id: i for i in tasks.next_word_items(triples, lang="ur")}
    en_items = {i.triple_id: i for i in tasks.next_word_items(triples, lang="en")}
    shared_ids = [tid for tid in ur_items if tid in en_items]
    if not shared_ids:
        raise SystemExit("no content-matched ur/en pairs; try a larger --limit")

    probe = extraction.extract_hidden_states(model, tokenizer, ur_items[shared_ids[0]].prefix, device=args.device)
    num_layers = probe.layers.shape[0]
    layer = max(1, min(num_layers - 1, round(args.layer_frac * (num_layers - 1))))
    print(f"n={len(shared_ids)} pairs; patching at layer {layer}/{num_layers - 1}")

    deltas = []
    for tid in shared_ids:
        ur_prefix = ur_items[tid].prefix
        en_prefix = en_items[tid].prefix
        hs_en = extraction.extract_hidden_states(model, tokenizer, en_prefix, device=args.device)
        enc_ur = tokenizer(ur_prefix, return_tensors="pt").to(args.device)

        with torch.no_grad():
            baseline_logits = model(**enc_ur).logits[0, -1, :].float().cpu()
        baseline = logit_lens.layer_script_mass(
            baseline_logits, decode_id, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn
        )

        replacement = hs_en.layers[layer, -1, :]
        with causal.patch_hidden_state(model, layer=layer, position=-1, replacement=replacement):
            with torch.no_grad():
                patched_logits = model(**enc_ur).logits[0, -1, :].float().cpu()
        patched = logit_lens.layer_script_mass(
            patched_logits, decode_id, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn
        )
        deltas.append(patched["latin"] - baseline["latin"])

    mean_delta = sum(deltas) / len(deltas)
    print(f"mean Δ Latin-script mass (patched − baseline) = {mean_delta:+.4f}")
    print("Positive values mean the English patch increased Latin-script next-token mass.")


if __name__ == "__main__":
    main()
