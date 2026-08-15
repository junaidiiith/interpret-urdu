"""Deterministic, dictionary-free script classification by Unicode code
point. No LLM judge, no ML classifier -- a token either falls in a script's
Unicode block or it doesn't.

RQ2-3: "Latin script" is a Unicode-block fact, not a language -- Latin-script
mass can include Roman-Urdu texting, numerals-as-words, or other Latin-script
content, not just English. classify_latin_subtype below is a lightweight,
disclosed-as-approximate wordlist check for anyone who needs to specifically
claim "English" rather than "Latin-script"."""
from __future__ import annotations

import re

_URDU_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
_DEVANAGARI_RANGES = [(0x0900, 0x097F)]
_LATIN_RANGES = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]

SCRIPTS = ("urdu", "hindi", "latin", "other")

# A small, hardcoded seed of common English function/content words -- enough
# to catch the highest-frequency logit-lens top-k tokens (articles,
# pronouns, prepositions), not a general English lexicon.
COMMON_ENGLISH_WORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but",
        "if", "of", "to", "in", "on", "at", "for", "with", "as", "by", "from", "that", "this",
        "these", "those", "it", "he", "she", "they", "we", "you", "i", "not", "no", "yes",
        "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "should",
        "what", "when", "where", "who", "why", "how", "all", "some", "one", "two", "three",
        "new", "good", "time", "people", "world", "also", "more", "most", "very", "just",
        "about", "into", "than", "which", "there", "their", "them", "his", "her", "its", "my",
        "your", "our", "so", "up", "out", "over", "then", "now", "here", "get", "make", "go",
        "know", "see", "say", "said", "like", "us", "am", "may", "must", "shall",
    }
)


def classify_latin_subtype(
    token: str, roman_urdu_words: frozenset[str], english_words: frozenset[str] = COMMON_ENGLISH_WORDS
) -> str:
    """Disambiguate a Latin-script token as 'english', 'roman_urdu', or
    'ambiguous' by wordlist membership -- deliberately simple (RESOLUTIONS.md
    RQ2-3 calls a wordlist check "enough"), not a language-ID model. A token
    matching both or neither list is 'ambiguous' rather than guessed at."""
    normalized = re.sub(r"[^a-z']", "", token.strip().lower())
    if not normalized:
        return "ambiguous"
    in_english = normalized in english_words
    in_roman_urdu = normalized in roman_urdu_words
    if in_english and not in_roman_urdu:
        return "english"
    if in_roman_urdu and not in_english:
        return "roman_urdu"
    return "ambiguous"


# RQ2 follow-up EXP-7: Persian and Urdu share the Perso-Arabic Unicode
# block, so classify_token/classify_char alone cannot distinguish them --
# unlike the Hindi (Devanagari) or English (Latin) competitors used
# elsewhere in RQ2, a same-script control needs a wordlist check at the
# level classify_latin_subtype already uses for English-vs-Roman-Urdu. Both
# lists are common, high-frequency closed-class function words (copulas,
# postpositions/prepositions, pronouns, negation) chosen specifically because
# their surface form differs between the two languages despite the shared
# script -- not a general lexicon, and it will leave most open-class content
# words (which are often genuinely shared/cognate between Persian and Urdu)
# unclassified as "neither", by design.
URDU_ARABIC_SCRIPT_MARKERS = frozenset(
    {
        "ہے", "ہیں", "ہو", "تھا", "تھی", "تھے", "کے", "کی", "کا", "کو", "سے", "میں", "پر",
        "یہ", "وہ", "کیا", "نہیں", "ہوگا", "ہوگی", "بہت", "لیے", "ساتھ", "رکھتا", "رکھتی",
        "سکتا", "سکتی", "والا", "والی", "گا", "گی", "گے", "اور", "اپنے", "اپنی", "جو", "کوئی",
    }
)
PERSIAN_MARKERS = frozenset(
    {
        "است", "هست", "بود", "شد", "شود", "می‌شود", "میشود", "برای", "این", "آن", "در", "با",
        "نیست", "می‌تواند", "میتواند", "دارد", "از", "به", "که", "چه", "خیلی", "هستند", "کرد",
        "نمی", "را", "هم", "آنها", "خود",
    }
)


def classify_arabic_script_subtype(token: str, urdu_words: frozenset[str] = URDU_ARABIC_SCRIPT_MARKERS, persian_words: frozenset[str] = PERSIAN_MARKERS) -> str:
    """Disambiguate a Perso-Arabic-script token as 'urdu', 'persian', or
    'neither' by closed-class wordlist membership (see the module-level
    comment above URDU_ARABIC_SCRIPT_MARKERS) -- deliberately narrow, the
    same kind of approximation classify_latin_subtype makes for English vs.
    Roman-Urdu, not a language-ID model. Most real content words will
    legitimately fall in 'neither', since Persian and Urdu share a large
    part of their vocabulary (this is exactly why EXP-7 targets closed-class
    function words instead of trying to classify every token)."""
    normalized = token.strip()
    if not normalized:
        return "neither"
    if normalized in urdu_words:
        return "urdu"
    if normalized in persian_words:
        return "persian"
    return "neither"


def classify_char(ch: str) -> str:
    cp = ord(ch)
    for lo, hi in _URDU_RANGES:
        if lo <= cp <= hi:
            return "urdu"
    for lo, hi in _DEVANAGARI_RANGES:
        if lo <= cp <= hi:
            return "hindi"
    for lo, hi in _LATIN_RANGES:
        if lo <= cp <= hi:
            return "latin"
    return "other"


def classify_token(token: str) -> str:
    """Majority vote over the token's characters, restricted to characters
    that belong to one of the three tracked scripts (digits/punctuation/space
    don't vote). A token with no script-bearing character classifies as
    'other'."""
    counts = {"urdu": 0, "hindi": 0, "latin": 0}
    for ch in token:
        c = classify_char(ch)
        if c in counts:
            counts[c] += 1
    if not any(counts.values()):
        return "other"
    return max(counts, key=counts.get)
