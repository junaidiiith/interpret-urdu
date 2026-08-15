"""Aggregates per-model RQ1/RQ2/RQ3 CSVs into the cross-model summaries the
writeup needs: replication (direction-match fraction), the RQ1-RQ2 joint
check, RQ2-6's continuous crossover statistic, and RQ3's BH-corrected
significance -- run this last, after every per-model script has produced its
output.

    python scripts/analyze_all.py
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd
from scipy import stats as scipy_stats

from urdu_interp import corpora, paths, stats
from urdu_interp.reporting import save_table

_LENGTH_LOADERS = {
    "rekhta_ghazals": lambda: corpora.load_rekhta("ghazals"),
    "rekhta_nazms": lambda: corpora.load_rekhta("nazms"),
    "flores_dev": lambda: corpora.load_flores("dev"),
    "ebible": lambda: corpora.load_ebible(),
}


def load_concat(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f"no files matched {pattern} -- run the per-model scripts first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def rq1_summary(df: pd.DataFrame) -> pd.DataFrame:
    # peak_layer_frac / effect_size_auc are the primary stats (RQ1-5) --
    # recovery_layer saturates at 0-1 for essentially every combination in
    # this project's real runs and is kept only as a secondary diagnostic.
    return (
        df.groupby(["model", "corpus", "lang_pair"])
        .agg(
            peak_layer_frac=("peak_layer_frac", "first"),
            effect_size_auc=("effect_size_auc", "first"),
            max_effect_size=("effect_size", "max"),
            recovery_layer=("recovery_layer", "first"),
            n_layers=("layer", "max"),
        )
        .reset_index()
    )


def rq2_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "corpus", "lang", "triple_id"])
        .agg(native_crossover_layer=("native_crossover_layer", "first"))
        .reset_index()
        .groupby(["model", "corpus", "lang"])["native_crossover_layer"]
        .median()
        .reset_index()
        .rename(columns={"native_crossover_layer": "median_crossover_layer"})
    )


def rq1_rq2_correlation(rq1: pd.DataFrame, rq2: pd.DataFrame, lang_pair: str, lang: str) -> None:
    # STA-3: average RQ1 across corpora per model first (one point per
    # model, not one per model x corpus) before correlating against RQ2's
    # single per-model crossover number. The models here span only two or
    # three distinct network depths (Qwen's 25 layers vs SmolLM3/Gemma's
    # 30-somethings), so a large share of any correlation is plausibly just
    # "deeper architecture -> both metrics' layer indices are numerically
    # larger," not a validated link between the two mechanisms -- not a
    # significance claim regardless of which p-value comes out.
    rq1_per_model = rq1[rq1["lang_pair"] == lang_pair].groupby("model")["peak_layer_frac"].mean().reset_index()
    rq2_per_model = rq2[rq2["lang"] == lang][["model", "median_crossover_layer"]]
    merged = rq1_per_model.merge(rq2_per_model, on="model")
    if len(merged) < 3:
        print(f"not enough overlapping models ({len(merged)}) for a correlation between RQ1 and RQ2")
        return
    rho, p = scipy_stats.spearmanr(merged["peak_layer_frac"], merged["median_crossover_layer"])
    print(f"RQ1 peak_layer_frac vs RQ2 median crossover layer ({lang_pair}/{lang}, "
          f"n={len(merged)} models, one point per model averaged across corpora): rho={rho:.3f}")
    print(f"  n={len(merged)} is too small to support a significance claim either way, and the models span "
          f"only a couple distinct network depths -- treat this as a scatter plot with a likely "
          f"architecture-depth confound, not a validated cross-check (raw p-value for reference only: p={p:.4f}).")


def rq2_continuous_gap(rq2_df: pd.DataFrame) -> pd.DataFrame:
    """RQ2-6: at ~90% relative network depth, the per-item target-minus-
    max-competitor script-mass gap, permutation-tested between the ur and hi
    conditions -- a continuous alternative to the discrete crossover-layer
    index (RQ2-6/RES-3), which is only precise to +/-1 layer and can't by
    itself distinguish "real gap" from "resolution floor of the metric"."""
    rows = []
    for (model, corpus), g in rq2_df.groupby(["model", "corpus"]):
        n_layers = int(g["layer"].max()) + 1
        target_layer = round(0.9 * (n_layers - 1))
        gaps = {}
        for lang, target_col, competitor_cols in (
            ("ur", "mass_urdu", ["mass_hindi", "mass_latin"]),
            ("hi", "mass_hindi", ["mass_urdu", "mass_latin"]),
        ):
            sub = g[(g["lang"] == lang) & (g["layer"] == target_layer)]
            if sub.empty or target_col not in sub.columns:
                continue
            gaps[lang] = (sub[target_col] - sub[competitor_cols].max(axis=1)).tolist()
        if gaps.get("ur") and gaps.get("hi"):
            observed, p = stats.permutation_test_diff_of_means(gaps["ur"], gaps["hi"])
            rows.append(
                {
                    "model": model,
                    "corpus": corpus,
                    "layer": target_layer,
                    "n_layers": n_layers,
                    "ur_minus_hi_gap_diff": observed,
                    "p_value": p,
                }
            )
    return pd.DataFrame(rows)


def dat5_length_covariate(rq1_df: pd.DataFrame) -> None:
    """DAT-5: corpora that differ in peak_layer_frac might just differ in
    sentence/line length (eBible sentences are shorter than ghazal couplets
    by construction), not "register" -- partial-correlate peak_layer_frac
    against each corpus's mean word count, controlling for model by
    within-model centering (subtracting each model's own across-corpus mean
    peak_layer_frac) before correlating against length. A strong surviving
    correlation here means the "register" story can't be cleanly
    distinguished from a simple length effect."""
    lengths = {}
    for corpus_name in rq1_df["corpus"].unique():
        loader = _LENGTH_LOADERS.get(corpus_name)
        if loader is None:
            continue
        triples = loader()
        lengths[corpus_name] = sum(len(t.ur.split()) for t in triples) / len(triples)
    if len(lengths) < 2:
        print("\n(skipping DAT-5: fewer than 2 corpora with a known length loader)")
        return

    df = rq1_df[rq1_df["corpus"].isin(lengths)].copy()
    df["mean_word_count"] = df["corpus"].map(lengths)
    df["peak_layer_frac_within_model"] = df["peak_layer_frac"] - df.groupby("model")["peak_layer_frac"].transform("mean")
    rho, p = scipy_stats.spearmanr(df["mean_word_count"], df["peak_layer_frac_within_model"])
    print("\n=== DAT-5: peak_layer_frac vs. corpus mean word count (within-model, model partialed out) ===")
    print(f"corpus mean word counts: { {k: round(v, 1) for k, v in lengths.items()} }")
    print(f"Spearman rho={rho:.3f}, p={p:.4f}, n={len(df)} rows (model x corpus x lang_pair)")


def rq3_significance(glob_pattern: str) -> pd.DataFrame:
    """Recompute both the (pseudoreplicated) Wilcoxon and the
    autocorrelation-aware block-bootstrap paired test directly from each
    saved RQ3 before/after CSV, instead of relying on run_rq3_eval.py's
    stdout -- keeps this aggregation self-contained and re-runnable."""
    rows = []
    for f in glob.glob(glob_pattern):
        if "__check_" in f:
            # RQ3-4's hyperparameter-grid reruns and other sensitivity
            # checks are saved with this marker specifically so they don't
            # get folded into the canonical per-model significance table.
            continue
        df = pd.read_csv(f)
        if "stage" not in df.columns:
            continue
        before = df[df["stage"] == "before"].sort_values("layer")["effect_size"].tolist()
        after = df[df["stage"] == "after"].sort_values("layer")["effect_size"].tolist()
        if len(before) != len(after) or len(before) < 2:
            continue
        model = str(df["model"].iloc[0]).split("[")[0]
        _, wilcoxon_p = stats.paired_delta_test(before, after)
        boot_delta, boot_p = stats.block_bootstrap_paired_test(before, after)
        rows.append(
            {
                "model": model,
                "file": f,
                "wilcoxon_p_not_autocorrelation_aware": wilcoxon_p,
                "block_bootstrap_mean_delta": boot_delta,
                "block_bootstrap_p_preferred": boot_p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq1-glob", default=str(paths.RQ1_DIR / "*.csv"))
    parser.add_argument("--rq2-glob", default=str(paths.RQ2_DIR / "*.csv"))
    parser.add_argument("--rq3-glob", default=str(paths.RQ3_DIR / "*.csv"))
    parser.add_argument("--lang-pair", default="ur-hi")
    parser.add_argument("--lang", default="ur")
    args = parser.parse_args()

    paths.ensure_dirs()

    # Only the per-model whole-CSV files (not the RQ1-1 fragmentation-
    # correlation side files, which have a different schema).
    # Exclude the fragmentation-correlation side files (different schema) and
    # any sensitivity/isolation-check rerun (RQ1-6, ENG-1, ...), which are
    # saved with a "__check_" marker via --output-suffix specifically so
    # they don't get double-counted here alongside the canonical run.
    rq1_files = [f for f in glob.glob(args.rq1_glob) if "fragmentation_corr" not in f and "__check_" not in f]
    if not rq1_files:
        raise SystemExit(f"no files matched {args.rq1_glob} -- run run_rq1_convergence.py first")
    rq1_df = pd.concat([pd.read_csv(f) for f in rq1_files], ignore_index=True)
    rq1 = rq1_summary(rq1_df)
    save_table(rq1, paths.RESULTS_DIR / "rq1_summary.csv")
    print("=== RQ1: peak_layer_frac / effect_size_auc per model/corpus/lang-pair ===")
    print(rq1.to_string(index=False))

    replication = rq1[rq1["lang_pair"] == args.lang_pair]
    direction_match = (replication["recovery_layer"].notna()).mean()
    print(f"\nRQ1 direction-match fraction ({args.lang_pair}, gap resolves at all, secondary diagnostic): "
          f"{direction_match:.2f} ({replication['recovery_layer'].notna().sum()}/{len(replication)} models)")

    dat5_length_covariate(rq1)

    rq2_continuous = pd.DataFrame()
    try:
        # Raw per-lens, canonical top_k=50 rows only -- skip the __tunedlens
        # (RQ2-1) and __topk (RQ2-4) sensitivity-rerun suffix files so the
        # canonical summary isn't silently mixed with those reruns.
        rq2_files = [f for f in glob.glob(args.rq2_glob) if "tunedlens" not in f and "__topk" not in f and "multiposition" not in f]
        if not rq2_files:
            raise SystemExit(f"no files matched {args.rq2_glob}")
        rq2_df = pd.concat([pd.read_csv(f) for f in rq2_files], ignore_index=True)
        rq2 = rq2_summary(rq2_df)
        save_table(rq2, paths.RESULTS_DIR / "rq2_summary.csv")
        print("\n=== RQ2: median native-crossover layer per model/corpus/lang ===")
        print(rq2.to_string(index=False))
        print()
        rq1_rq2_correlation(rq1, rq2, args.lang_pair, args.lang)

        rq2_continuous = rq2_continuous_gap(rq2_df)
    except SystemExit as e:
        print(f"\n(skipping RQ2 summary: {e})")

    rq3_sig = pd.DataFrame()
    rq3_files = glob.glob(args.rq3_glob)
    if rq3_files:
        rq3_sig = rq3_significance(args.rq3_glob)

    # STA-2: BH-correct across every p-value produced by this run (RQ3's
    # paired tests + RQ2-6's permutation tests) instead of reading each one
    # in isolation -- this was written and unit-tested in stats.py but never
    # called by any script before this.
    all_p: list[float] = []
    if not rq3_sig.empty:
        all_p.extend(rq3_sig["block_bootstrap_p_preferred"].tolist())
    if not rq2_continuous.empty:
        all_p.extend(rq2_continuous["p_value"].tolist())
    if all_p:
        q_values = stats.benjamini_hochberg(all_p)
        if not rq3_sig.empty:
            rq3_sig["q_value"] = q_values[: len(rq3_sig)]
            q_values = q_values[len(rq3_sig):]
        if not rq2_continuous.empty:
            rq2_continuous["q_value"] = q_values[: len(rq2_continuous)]

    if not rq3_sig.empty:
        save_table(rq3_sig, paths.RESULTS_DIR / "rq3_significance.csv")
        print("\n=== RQ3 significance (block-bootstrap preferred over Wilcoxon; BH q-values) ===")
        print(rq3_sig.to_string(index=False))

    if not rq2_continuous.empty:
        save_table(rq2_continuous, paths.RESULTS_DIR / "rq2_continuous_gap.csv")
        print("\n=== RQ2-6: continuous target-minus-competitor script-mass gap at ~90% depth (BH q-values) ===")
        print(rq2_continuous.to_string(index=False))


if __name__ == "__main__":
    main()
