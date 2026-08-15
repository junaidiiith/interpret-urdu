"""RQ2-5: is the crossover-layer pattern stable across the first few
*generated* token positions, not just the prompt's single final position?
The rest of RQ2 measures one next-token snapshot per item -- the thinnest
possible slice of evidence for a claim about a network-depth "detour".
Greedily decodes `--n-steps` tokens and runs the logit-lens detour at each
one (pipeline.run_detour_over_generation), reporting the median crossover
layer per step.

    python scripts/run_rq2_multiposition_detour.py --model qwen3.5-0.8b-base \\
        --corpus ebible --lang ur --limit 100 --n-steps 5 --device cuda:0
"""
from __future__ import annotations

import argparse

from urdu_interp import corpora, models, paths, pipeline, reporting, script_utils, tasks
from urdu_interp.logit_lens import make_unembed_fn

SCRIPT_BY_LANG = {"ur": "urdu", "hi": "hindi", "en": "latin"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--corpus", default="ebible", choices=["ebible", "flores_dev"])
    parser.add_argument("--lang", required=True, choices=list(SCRIPT_BY_LANG))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    unembed_fn = make_unembed_fn(model)
    roman_urdu_words = corpora.load_roman_urdu_wordlist()
    latin_subtype_fn = lambda token: script_utils.classify_latin_subtype(token, roman_urdu_words)  # noqa: E731

    triples = corpora.load_ebible(limit=args.limit) if args.corpus == "ebible" else corpora.load_flores("dev", limit=args.limit)
    items = tasks.next_word_items(triples, lang=args.lang)
    print(f"{len(items)} items from {args.corpus}/{args.lang}")

    target = SCRIPT_BY_LANG[args.lang]
    competitors = [s for s in ("urdu", "hindi", "latin") if s != target]

    rows = []
    crossover_by_step: dict[int, list[int]] = {step: [] for step in range(args.n_steps)}
    for item in items:
        detours = pipeline.run_detour_over_generation(
            model, tokenizer, item, unembed_fn, target, competitors, args.device,
            n_steps=args.n_steps, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn,
        )
        for step, detour in enumerate(detours):
            if detour.native_crossover_layer is not None:
                crossover_by_step[step].append(detour.native_crossover_layer)
            rows.append(
                {
                    "model": args.model,
                    "corpus": args.corpus,
                    "lang": args.lang,
                    "triple_id": item.triple_id,
                    "step": step,
                    "native_crossover_layer": detour.native_crossover_layer,
                }
            )

    for step in range(args.n_steps):
        vals = crossover_by_step[step]
        if vals:
            median = sorted(vals)[len(vals) // 2]
            print(f"step {step}: median crossover layer={median}, resolved {len(vals)}/{len(items)}")
        else:
            print(f"step {step}: no item resolved a stable crossover")

    df = reporting.rows_to_dataframe(rows)
    out_path = paths.RQ2_DIR / f"{args.model}__{args.corpus}__{args.lang}__multiposition.csv"
    reporting.save_table(df, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
