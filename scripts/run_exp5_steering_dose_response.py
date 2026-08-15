"""RQ2 follow-up EXP-5: same causal question as EXP-4 (does a decodable
English-leaning representation actually influence real output?), but
graded: build an "English direction" at one layer as the difference of means
between English-context and Urdu-context hidden states (over content-matched
FLORES triples, no new data), then add alpha*direction to that layer's
residual stream throughout real generation, for a sweep of alpha, and score
the *actual generated text* with the same word-level script classifier
EXP-6 uses. A monotonic dose-response curve (more nudge -> more English
leakage in real output) is stronger causal evidence than a single
patch/no-patch comparison.

Scope: Qwen3.5 and SmolLM3 only (see causal.py).

    python scripts/run_exp5_steering_dose_response.py --model qwen3.5-0.8b-instruct --device cuda:2
"""
from __future__ import annotations

import argparse

import torch

from urdu_interp import causal, corpora, extraction, models, paths, reporting, script_utils, tasks

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def build_direction(model, tokenizer, triples, layer: int, device: str, n_pairs: int = 60) -> torch.Tensor:
    ur_vecs, en_vecs = [], []
    for t in triples[:n_pairs]:
        if not t.ur or not t.en:
            continue
        hs_ur = extraction.extract_hidden_states(model, tokenizer, t.ur, device=device)
        hs_en = extraction.extract_hidden_states(model, tokenizer, t.en, device=device)
        ur_vecs.append(extraction.sentence_vector(hs_ur)[layer])
        en_vecs.append(extraction.sentence_vector(hs_en)[layer])
    ur_mean = torch.stack(ur_vecs).mean(dim=0)
    en_mean = torch.stack(en_vecs).mean(dim=0)
    direction = en_mean - ur_mean
    return direction / direction.norm()


def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def _code_switch_rate(text: str) -> float:
    words = text.split()
    counts = {"urdu": 0, "hindi": 0, "latin": 0, "other": 0}
    for w in words:
        counts[script_utils.classify_token(w)] += 1
    scripted = counts["urdu"] + counts["hindi"] + counts["latin"]
    return (counts["hindi"] + counts["latin"]) / scripted if scripted else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer-frac", type=float, default=0.5, help="fraction of network depth to steer at")
    parser.add_argument("--n-prompts", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fit_triples = corpora.load_flores("dev", limit=80)
    probe_hs = extraction.extract_hidden_states(model, tokenizer, fit_triples[0].ur, device=args.device)
    num_layers = probe_hs.layers.shape[0]
    layer = max(1, round(args.layer_frac * (num_layers - 1)))
    print(f"[{args.model}] building English-minus-Urdu direction at layer {layer}/{num_layers}...")
    direction = build_direction(model, tokenizer, fit_triples, layer, args.device)

    eval_triples = corpora.load_ebible(limit=args.n_prompts)
    ur_prompts = [t.ur for t in eval_triples if t.ur][: args.n_prompts]

    rows = []
    for alpha in ALPHAS:
        for i, prompt in enumerate(ur_prompts):
            if alpha == 0.0:
                text = _generate(model, tokenizer, prompt, args.device, args.max_new_tokens)
            else:
                with causal.steer_hidden_state(model, layer=layer, direction=direction, alpha=alpha):
                    text = _generate(model, tokenizer, prompt, args.device, args.max_new_tokens)
            rate = _code_switch_rate(text)
            rows.append({"model": args.model, "layer": layer, "num_layers": num_layers, "alpha": alpha, "prompt_idx": i, "generated": text, "code_switch_rate": rate})
        rates = [r["code_switch_rate"] for r in rows if r["alpha"] == alpha and r["code_switch_rate"] == r["code_switch_rate"]]
        mean_rate = sum(rates) / len(rates) if rates else float("nan")
        print(f"[{args.model}] alpha={alpha}: mean code_switch_rate={mean_rate:.3f} over {len(rates)} generations")

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RESULTS_DIR / "exp5_steering_dose_response" / f"{args.model}.csv"
    reporting.save_table(df, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
