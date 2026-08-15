"""RQ1: does a model's Urdu representation converge, layer by layer, with its
Hindi (and English) counterpart on matched content -- and if so, how early?

    python scripts/run_rq1_convergence.py --model qwen3.5-0.8b-base \\
        --corpus rekhta_ghazals --lang-pairs ur-hi ur-en --limit 300
"""
from __future__ import annotations

import argparse

from scipy import stats as scipy_stats

from urdu_interp import convergence, corpora, models, paths, pipeline, reporting

CORPUS_LOADERS = {
    "rekhta_ghazals": lambda limit: corpora.load_rekhta("ghazals", limit=limit),
    "rekhta_nazms": lambda limit: corpora.load_rekhta("nazms", limit=limit),
    "flores_dev": lambda limit: corpora.load_flores("dev", limit=limit),
    "ebible": lambda limit: corpora.load_ebible(limit=limit),
    # DAT-3: a second eBible translation triple, for a replication check
    # against the default urdgvu/hincv/eng-web triple.
    "ebible_alt": lambda limit: corpora.load_ebible(
        ur_translation="urdoucv", hi_translation="hin2017", en_translation="eng-kjv2006", limit=limit
    ),
}
# Rekhta's ur/hi/en-rm fields are transliterations (word order preserved), so
# word-level alignment is exact; the other corpora are independent
# translations, compared as whole sentences instead (see alignment.py).
WORD_LEVEL_CORPORA = {"rekhta_ghazals", "rekhta_nazms"}
# RQ1-2: word-level (Rekhta) and sentence-level (everything else) peak
# scores have different baseline statistics (sentence vectors average over
# more tokens) and are not directly comparable -- every row is tagged with
# its granularity so downstream analysis can't silently conflate them.
# MOT-4/DAT-6: Rekhta's "en" field is en-rm, a same-content Roman
# transliteration of the Urdu text, not an independently authored English
# translation the way FLORES/eBible's "en" is -- tagged as content_type so
# the "English beats Hindi" comparison doesn't quietly average across two
# different kinds of "English".
CONTENT_TYPE = {corpus: ("transliteration" if corpus in WORD_LEVEL_CORPORA else "translation") for corpus in CORPUS_LOADERS}


def _fragmentation_correlation(fragmentation: list[int], per_item_scores) -> list[tuple[int, float, float]]:
    """RQ1-1: Spearman-correlate per-word tokenizer fragmentation against
    that word's per-item convergence score, at every layer -- the check this
    pipeline stage (pipeline.fragmentation_covariate) was built for and never
    ran until now. A strong early-layer correlation that vanishes in deep
    layers would support "early gap is a tokenization artifact"; a
    correlation that persists deep would argue against that reading."""
    num_layers = per_item_scores.shape[1]
    rows = []
    for layer in range(num_layers):
        rho, p = scipy_stats.spearmanr(fragmentation, per_item_scores[:, layer].tolist())
        rows.append((layer, float(rho), float(p)))
    return rows


def _run_word_level(args, model, tokenizer, triples) -> list[dict]:
    # Extract every requested lang-pair's word batch up front so lang-pairs
    # sharing the same lang_a (e.g. "ur" in both ur-hi and ur-en) can share a
    # centering population (RQ1-3) instead of each computing its own mean
    # from its own sub-batch -- otherwise the "same" Urdu vectors center
    # differently depending on which language they're being compared to,
    # which silently un-levels the ur-hi vs ur-en comparison.
    batches = {}
    for lang_pair in args.lang_pairs:
        lang_a, lang_b = lang_pair.split("-")
        batch = pipeline.extract_triple_word_vectors(model, tokenizer, triples, lang_a, lang_b, args.device)
        batches[lang_pair] = (lang_a, lang_b, batch)
        print(f"{lang_pair}: {batch.stats}")

    by_lang_a: dict[str, list[str]] = {}
    for lang_pair, (lang_a, _, _) in batches.items():
        by_lang_a.setdefault(lang_a, []).append(lang_pair)

    shared_means = {}
    for lang_a, pairs in by_lang_a.items():
        union_batches = [(batches[lp][2].vecs_a, batches[lp][2].ids) for lp in pairs]
        shared_means[lang_a] = convergence.union_layer_mean(union_batches)
        if len(pairs) > 1:
            for lp in pairs:
                vecs_a = batches[lp][2].vecs_a
                old_mean = vecs_a.mean(dim=0)
                gap = (old_mean - shared_means[lang_a]).norm(dim=-1).mean().item()
                print(f"[RQ1-3] {lp}: mean per-layer L2 gap between the old per-batch '{lang_a}' centering mean and the new shared union mean = {gap:.4f}")

    all_rows: list[dict] = []
    for lang_pair in args.lang_pairs:
        lang_a, lang_b, batch = batches[lang_pair]
        result = convergence.layerwise_convergence(
            batch.vecs_a, batch.vecs_b, n_null=args.n_null, mean_a=shared_means[lang_a]
        )
        recovery = convergence.recovery_layer(result.effect_size, threshold=args.recovery_threshold)
        peak_frac = convergence.peak_layer_frac(result.layer_scores)
        print(f"{lang_pair}: recovery_layer={recovery} of {len(result.effect_size)} layers, "
              f"max effect_size={max(result.effect_size):.2f}, peak_layer_frac={peak_frac:.3f}")

        rows = reporting.convergence_rows(args.model, args.corpus, lang_pair, result)
        for row in rows:
            row["recovery_layer"] = recovery
            row["granularity"] = "word"
            row["content_type"] = CONTENT_TYPE[args.corpus]
        all_rows.extend(rows)

        frag_rows = _fragmentation_correlation(batch.fragmentation_a, result.per_item_scores)
        frag_df = reporting.rows_to_dataframe(
            [
                {
                    "model": args.model,
                    "corpus": args.corpus,
                    "lang_pair": lang_pair,
                    "layer": layer,
                    "fragmentation_spearman_rho": rho,
                    "fragmentation_spearman_p": p,
                }
                for layer, rho, p in frag_rows
            ]
        )
        frag_path = paths.RQ1_DIR / f"{args.model}__{args.corpus}__{lang_pair}__fragmentation_corr{args.output_suffix}.csv"
        reporting.save_table(frag_df, frag_path)
        half = len(frag_rows) // 2
        early_rho = [r for _, r, _ in frag_rows[:half]]
        late_rho = [r for _, r, _ in frag_rows[half:]]
        print(f"{lang_pair}: fragmentation-vs-convergence Spearman rho -- "
              f"early-half mean={sum(early_rho) / len(early_rho):.3f}, "
              f"late-half mean={sum(late_rho) / len(late_rho):.3f} (saved {frag_path})")

    return all_rows


def _run_sentence_level(args, model, tokenizer, triples) -> list[dict]:
    all_rows: list[dict] = []
    for lang_pair in args.lang_pairs:
        lang_a, lang_b = lang_pair.split("-")
        vecs_a, vecs_b, kept_ids = pipeline.extract_triple_sentence_vectors(model, tokenizer, triples, lang_a, lang_b, args.device)
        print(f"{lang_pair}: {len(kept_ids)} sentence pairs")

        result = convergence.layerwise_convergence(vecs_a, vecs_b, n_null=args.n_null)
        recovery = convergence.recovery_layer(result.effect_size, threshold=args.recovery_threshold)
        print(f"{lang_pair}: recovery_layer={recovery} of {len(result.effect_size)} layers, max effect_size={max(result.effect_size):.2f}")

        rows = reporting.convergence_rows(args.model, args.corpus, lang_pair, result)
        for row in rows:
            row["recovery_layer"] = recovery
            row["granularity"] = "sentence"
            row["content_type"] = CONTENT_TYPE[args.corpus]
        all_rows.extend(rows)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--corpus", required=True, choices=list(CORPUS_LOADERS))
    parser.add_argument("--lang-pairs", nargs="+", default=["ur-hi", "ur-en"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--recovery-threshold", type=float, default=0.5)
    parser.add_argument("--output-suffix", default="", help="appended to the output filename -- use for sensitivity/isolation reruns (e.g. RQ1-6, ENG-1) so they don't overwrite the canonical run")
    args = parser.parse_args()

    paths.ensure_dirs()
    spec = models.get_spec(args.model)
    model, tokenizer = models.load_model_and_tokenizer(spec, device=args.device)
    triples = CORPUS_LOADERS[args.corpus](args.limit)

    if args.corpus in WORD_LEVEL_CORPORA:
        all_rows = _run_word_level(args, model, tokenizer, triples)
    else:
        all_rows = _run_sentence_level(args, model, tokenizer, triples)

    df = reporting.rows_to_dataframe(all_rows)
    out_path = paths.RQ1_DIR / f"{args.model}__{args.corpus}{args.output_suffix}.csv"
    reporting.save_table(df, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
