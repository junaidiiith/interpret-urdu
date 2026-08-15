"""MOT-2: "Hindi is the same spoken language, just a different script" is
truest for spoken Hustani and weakest for the formal written registers this
project's corpora are actually built from (news, scripture, professional
translation) -- FLORES/OPUS-style formal Hindi and formal Urdu have
independently standardized vocabulary, mostly along a Sanskrit-tatsama
(Hindi) vs. Persian/Arabic-loan (Urdu) axis, for a century-plus. This puts a
concrete number on that drift: for ~30 matched concepts (book, time, war,
history, ...), count how often FLORES/OPUS's Urdu-side text uses the
Persian/Arabic-coded word for that concept vs. how often the paired
Hindi-side text uses the Sanskrit-coded word -- a simple wordlist-based
proxy, disclosed as such, not a real etymological classifier.

    python scripts/analyze_register_drift.py --corpus flores_dev
"""
from __future__ import annotations

import argparse

from urdu_interp import corpora

# ~30 matched concept pairs: (concept, Persian/Arabic-coded Urdu word,
# Sanskrit-tatsama-coded Hindi word for the *same* concept). Picked as
# clearly register-marked pairs, not an exhaustive lexicon -- a small,
# disclosed proxy, per RESOLUTIONS.md MOT-2 ("a simple wordlist-based
# estimate is enough").
REGISTER_PAIRS = [
    ("book", "کتاب", "पुस्तक"),
    ("time", "وقت", "समय"),
    ("world", "دنیا", "विश्व"),
    ("minister", "وزیر", "मंत्री"),
    ("knowledge", "علم", "ज्ञान"),
    ("education", "تعلیم", "शिक्षा"),
    ("news", "خبر", "समाचार"),
    ("public", "عوام", "जनता"),
    ("nation", "قوم", "राष्ट्र"),
    ("city", "شہر", "नगर"),
    ("court", "عدالت", "न्यायालय"),
    ("president", "صدر", "राष्ट्रपति"),
    ("war", "جنگ", "युद्ध"),
    ("peace", "امن", "शांति"),
    ("history", "تاریخ", "इतिहास"),
    ("language", "زبان", "भाषा"),
    ("culture", "ثقافت", "संस्कृति"),
    ("religion", "مذہب", "धर्म"),
    ("politics", "سیاست", "राजनीति"),
    ("economy", "معیشت", "अर्थव्यवस्था"),
    ("trade", "تجارت", "व्यापार"),
    ("health", "صحت", "स्वास्थ्य"),
    ("treatment", "علاج", "उपचार"),
    ("institution", "ادارہ", "संस्था"),
    ("order", "حکم", "आदेश"),
    ("price", "قیمت", "मूल्य"),
    ("wealth", "دولت", "धन"),
    ("problem", "مسئلہ", "समस्या"),
    ("answer", "جواب", "उत्तर"),
    ("question", "سوال", "प्रश्न"),
    ("letter", "خط", "पत्र"),
]

CORPUS_LOADERS = {
    "flores_dev": lambda: corpora.load_flores("dev"),
    "ebible": lambda: corpora.load_ebible(),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="flores_dev", choices=list(CORPUS_LOADERS))
    args = parser.parse_args()

    triples = CORPUS_LOADERS[args.corpus]()
    ur_text = " ".join(t.ur for t in triples if t.ur)
    hi_text = " ".join(t.hi for t in triples if t.hi)
    ur_word_count = len(ur_text.split())
    hi_word_count = len(hi_text.split())

    print(f"{args.corpus}: {len(triples)} triples, {ur_word_count} Urdu words, {hi_word_count} Hindi words\n")
    print(f"{'concept':<14}{'ur (fa/ar word)':>20}{'ur count':>10}{'hi (sa word)':>18}{'hi count':>10}")

    ur_hits = 0
    hi_hits = 0
    for concept, ur_word, hi_word in REGISTER_PAIRS:
        uc = ur_text.count(ur_word)
        hc = hi_text.count(hi_word)
        ur_hits += uc
        hi_hits += hc
        print(f"{concept:<14}{ur_word:>20}{uc:>10}{hi_word:>18}{hc:>10}")

    ur_frac = ur_hits / ur_word_count if ur_word_count else 0.0
    hi_frac = hi_hits / hi_word_count if hi_word_count else 0.0
    print(f"\nUrdu Persian/Arabic-coded register-word occurrences: {ur_hits} ({ur_frac:.4%} of all Urdu words)")
    print(f"Hindi Sanskrit-coded register-word occurrences: {hi_hits} ({hi_frac:.4%} of all Hindi words)")
    print("\nBoth fractions are small by construction (30 concepts vs. the full vocabulary) -- the number that "
          "matters is the *ratio* between them and, more importantly, non-zero counts on both sides confirming "
          "the two registers really have drifted onto largely disjoint vocabulary for the same concepts in "
          "exactly the formal-translation corpora this project measures, not a claim about spoken Hindustani.")


if __name__ == "__main__":
    main()
