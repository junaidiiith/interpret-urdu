"""RQ2 follow-up EXP-7: same-script control. Persian shares Urdu's
Perso-Arabic script, so it can't be told apart from Urdu by
script_utils.classify_token the way Hindi (Devanagari) and English (Latin)
can -- this instead uses a closed-class function-word wordlist
(classify_arabic_script_subtype) to ask a narrower version of the same
question: while the model processes real Urdu text (from a genuine Urdu-
Persian parallel corpus, OPUS GlobalVoices pivoted via English -- see
corpora.load_opus_ur_fa_pivot), do intermediate layers' top-k next-token
predictions transiently favor Persian-specific function words over
Urdu-specific ones, the same "wobble before committing" pattern RQ2 measures
against Hindi/English? If the same instability shows up against a
competitor with no relationship to this project's Hindi/English motivation,
that's evidence the original "detour" is a general early-layer property, not
a Hindi/English-specific one.

Caveat, disclosed rather than glossed over: most real top-k tokens are
neither list's markers (open-class Persian/Urdu vocabulary overlaps heavily,
by design left unclassified) -- per-layer mass here is a much smaller,
noisier residual than the full-vocabulary script-mass statistic RQ2 uses
elsewhere. resolved_frac reports how often there's any marker signal at all.

    python scripts/run_exp7_persian_control.py --model qwen3.5-0.8b-base --device cuda:2
"""
from __future__ import annotations

import argparse

from urdu_interp import corpora, extraction, models, paths, reporting, script_utils, tasks
from urdu_interp.logit_lens import make_unembed_fn
from urdu_interp.stats import first_stable_index

import torch


def layer_marker_mass(logits, decode_id, top_k: int = 50) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    k = min(top_k, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k)
    mass = {"urdu": 0.0, "persian": 0.0, "neither": 0.0}
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        token = decode_id(idx)
        mass[script_utils.classify_arabic_script_subtype(token)] += p
    return mass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--domain", default="globalvoices")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    unembed_fn = make_unembed_fn(model)

    def decode_id(tid: int) -> str:
        return tokenizer.decode([tid])

    triples = corpora.load_opus_ur_fa_pivot(args.domain)
    items = tasks.next_word_items(triples, lang="ur")
    print(f"[{args.model}] {len(items)} ur-fa parallel next-word items ({args.domain})")

    rows = []
    crossover_layers = []
    n_with_any_signal = 0
    for item in items:
        hs = extraction.extract_hidden_states(model, tokenizer, item.prefix, device=args.device)
        num_layers = hs.layers.shape[0]
        layer_masses = [layer_marker_mass(unembed_fn(hs.layers[l, -1, :]), decode_id, top_k=args.top_k) for l in range(num_layers)]
        any_signal = any(m["urdu"] > 0 or m["persian"] > 0 for m in layer_masses)
        n_with_any_signal += int(any_signal)
        flags = [m["urdu"] > m["persian"] for m in layer_masses]
        crossover = first_stable_index(flags)
        if crossover is not None:
            crossover_layers.append(crossover)
        for layer, m in enumerate(layer_masses):
            rows.append({"model": args.model, "triple_id": item.triple_id, "layer": layer, **{f"mass_{k}": v for k, v in m.items()}, "crossover_layer": crossover})

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RESULTS_DIR / "exp7_persian_control" / f"{args.model}.csv"
    reporting.save_table(df, out_path)

    print(f"\n[{args.model}] {n_with_any_signal}/{len(items)} items had any urdu/persian marker signal in top-{args.top_k} at some layer")
    if crossover_layers:
        median = sorted(crossover_layers)[len(crossover_layers) // 2]
        print(f"[{args.model}] median urdu-over-persian crossover layer={median}, resolved for {len(crossover_layers)}/{len(items)} items")
    else:
        print(f"[{args.model}] no item ever reached a stable urdu-marker-favoring crossover")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
