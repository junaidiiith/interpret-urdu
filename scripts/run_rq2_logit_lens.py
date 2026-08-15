"""RQ2: does the model transiently favor Hindi- or English-scripted tokens
before settling on the target language (the 'translation detour'), and how
deep into the network does that settling happen?

    python scripts/run_rq2_logit_lens.py --model qwen3.5-0.8b-base \\
        --corpus ebible --lang ur --limit 300
"""
from __future__ import annotations

import argparse

from urdu_interp import corpora, models, paths, pipeline, reporting, script_utils, tasks, tuned_lens
from urdu_interp.logit_lens import make_unembed_fn

CORPUS_LOADERS = {
    "ebible": lambda limit: corpora.load_ebible(limit=limit),
    "flores_dev": lambda limit: corpora.load_flores("dev", limit=limit),
    # DAT-6/RQ2-3 validation: genuine Roman-Urdu texting text (no hi/en
    # field) -- since every item here truly *is* Roman-Urdu, this is a real
    # ground-truth check of the English-vs-Roman-Urdu wordlist
    # disambiguation, not just Rekhta's literary transliteration.
    "roman_urdu_texting": lambda limit: corpora.load_roman_urdu_texting(limit=limit),
}
SCRIPT_BY_LANG = {"ur": "urdu", "hi": "hindi", "en": "latin"}
# roman_urdu_texting only populates the "ur" field, but its native script is
# Latin (it's Roman-Urdu, not Nastaliq Urdu) -- override the target/
# competitor split for that one corpus instead of the normal lang->script map.
CORPUS_TARGET_OVERRIDE = {"roman_urdu_texting": "latin"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--corpus", default="ebible", choices=list(CORPUS_LOADERS))
    parser.add_argument("--lang", required=True, choices=list(SCRIPT_BY_LANG))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--lens", default="raw", choices=["raw", "tuned"], help="RQ2-1: raw logit lens (default) or a fitted per-layer tuned lens")
    parser.add_argument("--lens-fit-corpus", default="flores_dev", help="corpus to sample ordinary text from when fitting the tuned lens")
    parser.add_argument("--lens-fit-size", type=int, default=200, help="number of sentences used to fit the tuned lens")
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)

    if args.lens == "raw":
        unembed_fn = make_unembed_fn(model)
    else:
        fit_triples = corpora.load_flores("dev", limit=args.lens_fit_size) if args.lens_fit_corpus == "flores_dev" else corpora.load_ebible(limit=args.lens_fit_size)
        fit_texts = [t.ur for t in fit_triples if t.ur] + [t.hi for t in fit_triples if t.hi] + [t.en for t in fit_triples if t.en]
        print(f"fitting tuned lens on {len(fit_texts)} texts from {args.lens_fit_corpus}...")
        lens_params = tuned_lens.fit_tuned_lens(model, tokenizer, fit_texts, args.device)
        unembed_fn = tuned_lens.make_tuned_unembed_fns(model, lens_params)

    roman_urdu_words = corpora.load_roman_urdu_wordlist()
    latin_subtype_fn = lambda token: script_utils.classify_latin_subtype(token, roman_urdu_words)  # noqa: E731

    triples = CORPUS_LOADERS[args.corpus](args.limit)
    items = tasks.next_word_items(triples, lang=args.lang)
    print(f"{len(items)} next-word items from {args.corpus}/{args.lang}")

    target = CORPUS_TARGET_OVERRIDE.get(args.corpus, SCRIPT_BY_LANG[args.lang])
    competitors = [s for s in ("urdu", "hindi", "latin") if s != target]
    results = pipeline.run_detour_for_items(
        model, tokenizer, items, unembed_fn, target, competitors, args.device, top_k=args.top_k, latin_subtype_fn=latin_subtype_fn
    )

    all_rows = []
    crossover_layers = []
    other_mass_by_layer: dict[int, list[float]] = {}
    for item, detour in results:
        if detour.native_crossover_layer is not None:
            crossover_layers.append(detour.native_crossover_layer)
        rows = reporting.detour_rows(args.model, args.corpus, args.lang, detour)
        for row in rows:
            row["triple_id"] = item.triple_id
            row["lens"] = args.lens
            other_mass_by_layer.setdefault(row["layer"], []).append(row["mass_other"])
        all_rows.extend(rows)

    if crossover_layers:
        median = sorted(crossover_layers)[len(crossover_layers) // 2]
        print(f"native crossover layer: median={median}, resolved for {len(crossover_layers)}/{len(results)} items")
    else:
        print("no item ever reached a stable native-script crossover")

    # RQ2-2: mass_other was computed but never looked at -- surface it here.
    n_low_confidence_layers = 0
    for layer in sorted(other_mass_by_layer):
        vals = other_mass_by_layer[layer]
        mean_other = sum(vals) / len(vals)
        if mean_other > 0.3:
            n_low_confidence_layers += 1
    print(f"mass_other: {n_low_confidence_layers}/{len(other_mass_by_layer)} layers have mean mass_other > 0.3 "
          f"(low-confidence: the urdu/hindi/latin-script split at those layers is read off a small residual)")

    df = reporting.rows_to_dataframe(all_rows)
    lens_suffix = "" if args.lens == "raw" else f"__{args.lens}lens"
    # RQ2-4: don't let a top_k sensitivity run silently overwrite the
    # canonical top_k=50 file for the same model/corpus/lang.
    topk_suffix = "" if args.top_k == 50 else f"__topk{args.top_k}"
    out_path = paths.RQ2_DIR / f"{args.model}__{args.corpus}__{args.lang}{lens_suffix}{topk_suffix}.csv"
    reporting.save_table(df, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
