"""Turns the dataclasses produced by convergence.py/logit_lens.py into flat
rows so scripts can build one DataFrame per RQ and save it uniformly."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from urdu_interp.convergence import ConvergenceResult, effect_size_auc, peak_layer_frac
from urdu_interp.logit_lens import DetourResult


def convergence_rows(model_key: str, corpus: str, lang_pair: str, result: ConvergenceResult) -> list[dict]:
    # peak_layer_frac/effect_size_auc are the primary summary stats (RQ1-5);
    # they're constant per (model, corpus, lang_pair) so every row carries
    # the same value, the same way recovery_layer already does.
    peak_frac = peak_layer_frac(result.layer_scores)
    auc = effect_size_auc(result.effect_size)
    return [
        {
            "model": model_key,
            "corpus": corpus,
            "lang_pair": lang_pair,
            "layer": layer,
            "n_pairs": result.n_pairs,
            "layer_score": result.layer_scores[layer],
            "null_mean": result.null_mean[layer],
            "null_std": result.null_std[layer],
            "effect_size": result.effect_size[layer],
            "peak_layer_frac": peak_frac,
            "effect_size_auc": auc,
        }
        for layer in range(len(result.layer_scores))
    ]


def detour_rows(model_key: str, corpus: str, lang: str, result: DetourResult) -> list[dict]:
    # RQ2-2: mass_other was computed but never surfaced -- if a large chunk
    # of top-k probability mass at a layer is punctuation/digits/undecodable
    # fragments, the urdu/hindi/latin split at that layer is being read off
    # a small, possibly noisy residual. low_confidence flags that directly
    # in the saved table instead of leaving it to be noticed by chance.
    return [
        {
            "model": model_key,
            "corpus": corpus,
            "lang": lang,
            "layer": layer,
            "native_crossover_layer": result.native_crossover_layer,
            **{f"mass_{script}": mass for script, mass in masses.items()},
            "low_confidence": masses.get("other", 0.0) > 0.3,
        }
        for layer, masses in enumerate(result.layer_masses)
    ]


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
