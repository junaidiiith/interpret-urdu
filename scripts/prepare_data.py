"""Builds and reports the matched-content datasets RQ1/RQ2 run on. No model is
loaded here -- this only exercises corpora.py + alignment.py, so it's fast and
safe to run before spending any GPU time.

    python scripts/prepare_data.py --limit 2000
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from urdu_interp import alignment, corpora, paths
from urdu_interp.reporting import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap per-corpus triples, for a quick check")
    args = parser.parse_args()

    paths.ensure_dirs()
    summary: dict[str, dict] = {}

    rekhta_ghazals = corpora.load_rekhta("ghazals", limit=args.limit)
    _, ur_hi_stats = alignment.align_words_positional(rekhta_ghazals, "ur", "hi")
    _, ur_en_stats = alignment.align_words_positional(rekhta_ghazals, "ur", "en")
    summary["rekhta_ghazals"] = {
        "n_triples": len(rekhta_ghazals),
        "ur_hi_word_alignment": asdict(ur_hi_stats),
        "ur_en_word_alignment": asdict(ur_en_stats),
    }

    rekhta_nazms = corpora.load_rekhta("nazms", limit=args.limit)
    summary["rekhta_nazms"] = {"n_triples": len(rekhta_nazms)}

    flores = corpora.load_flores("dev", limit=args.limit)
    summary["flores_dev"] = {"n_triples": len(flores)}

    ebible = corpora.load_ebible(limit=args.limit)
    summary["ebible"] = {"n_triples": len(ebible)}

    for domain in ("ted2020", "opensubtitles", "globalvoices", "qed"):
        opus = corpora.load_opus_pivot(domain, limit=args.limit)
        summary[f"opus_{domain}"] = {"n_triples": len(opus)}

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out_path = paths.DATA_CACHE_DIR / "corpus_summary.json"
    save_json(summary, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
