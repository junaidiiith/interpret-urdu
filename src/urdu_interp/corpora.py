"""Loaders that turn the raw files in data/ into AlignedTriple records.

Every loader here relies on an alignment that is exact by construction (shared
poem id, shared verse key, shared line number, or an exact-string pivot) --
never on a statistical aligner. See alignment.py for the word-level alignment
built on top of these triples.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from urdu_interp import paths


@dataclass(frozen=True)
class AlignedTriple:
    id: str
    source: str
    ur: str
    hi: str | None = None
    en: str | None = None
    # RQ2 follow-up EXP-7: Persian, same-script-family-as-Urdu control (see
    # load_opus_ur_fa_pivot) -- optional so every existing loader/call site
    # is unaffected.
    fa: str | None = None
    meta: dict = field(default_factory=dict)


def load_rekhta(kind: str = "ghazals", limit: int | None = None) -> list[AlignedTriple]:
    """Poem-level triples from the Rekhta corpus: ur (Nastaliq), hi (Devanagari
    transliteration of the *same words*), en (plain Roman transliteration).
    These are transliterations of one poem, not independent translations --
    word order and word count match 1:1 across the three fields.
    """
    fname = "rekhta_all_poets_ghazals.json" if kind == "ghazals" else "rekhta_all_poets_nazms.json"
    path = paths.POETRY_DIR / fname
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    triples: list[AlignedTriple] = []
    for poet_url, poems in data.items():
        if not isinstance(poems, dict):
            continue
        for poem_url, fields in poems.items():
            ur = fields.get("ur")
            hi = fields.get("hi")
            en = fields.get("en-rm")
            if not ur or not hi or not en:
                continue
            triples.append(
                AlignedTriple(
                    id=poem_url,
                    source=f"rekhta_{kind}",
                    ur=ur,
                    hi=hi,
                    en=en,
                    meta={"poet": poet_url},
                )
            )
            if limit is not None and len(triples) >= limit:
                return triples
    return triples


def load_flores(split: str = "dev", limit: int | None = None) -> list[AlignedTriple]:
    """Sentence-level triples from FLORES-200, aligned by line number (the
    files are professionally translated parallel releases, one sentence per
    line, already in lockstep across languages)."""
    base = paths.FLORES_DIR / split
    ur_lines = (base / "urd_Arab.dev").read_text(encoding="utf-8").splitlines() if split == "dev" else (base / f"urd_Arab.{split}").read_text(encoding="utf-8").splitlines()
    hi_lines = (base / "hin_Deva.dev").read_text(encoding="utf-8").splitlines() if split == "dev" else (base / f"hin_Deva.{split}").read_text(encoding="utf-8").splitlines()
    en_lines = (base / "eng_Latn.dev").read_text(encoding="utf-8").splitlines() if split == "dev" else (base / f"eng_Latn.{split}").read_text(encoding="utf-8").splitlines()

    n = min(len(ur_lines), len(hi_lines), len(en_lines))
    triples = [
        AlignedTriple(id=f"flores_{split}_{i}", source=f"flores_{split}", ur=ur_lines[i], hi=hi_lines[i], en=en_lines[i])
        for i in range(n)
    ]
    if limit is not None:
        triples = triples[:limit]
    return triples


def _parse_vpl(path: Path) -> dict[str, str]:
    """Parse an eBible verse-per-line file with an embedded 'BOOK CHAP:VERSE text' key."""
    verses: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        key = f"{parts[0]} {parts[1]}"
        text = parts[2]
        if text:
            verses[key] = text
    return verses


def load_ebible(
    ur_translation: str = "urdgvu",
    hi_translation: str = "hincv",
    en_translation: str = "eng-web",
    limit: int | None = None,
) -> list[AlignedTriple]:
    """Verse-level triples joined on the shared 'BOOK CHAP:VERSE' key -- an
    exact join, not a statistical one, since every eBible file is keyed this way."""
    ur = _parse_vpl(paths.EBIBLE_DIR / f"{ur_translation}_vpl.txt")
    hi = _parse_vpl(paths.EBIBLE_DIR / f"{hi_translation}_vpl.txt")
    en = _parse_vpl(paths.EBIBLE_DIR / f"{en_translation}_vpl.txt")

    shared_keys = sorted(set(ur) & set(hi) & set(en))
    if limit is not None:
        shared_keys = shared_keys[:limit]
    return [
        AlignedTriple(id=key, source="ebible", ur=ur[key], hi=hi[key], en=en[key])
        for key in shared_keys
    ]


_OPUS_DOMAINS = {"ted2020": "TED2020", "opensubtitles": "OpenSubtitles", "globalvoices": "GlobalVoices", "qed": "QED"}


def load_opus_pivot(domain: str = "ted2020", limit: int | None = None) -> list[AlignedTriple]:
    """Urdu/Hindi triples for one OPUS domain, joined via an exact-string pivot
    on the English side. OPUS's Urdu and Hindi sides are each aligned to
    English separately and are NOT the same set of English sentences, so a
    direct line-index join would be wrong -- this intersects the two English
    sides on exact string match instead, which is reliable (no statistical
    aligner) but yields a smaller, exact subset rather than the full corpus.
    """
    name = _OPUS_DOMAINS[domain.lower()]
    ur_en_path = paths.OPUS_UR_EN_DIR / f"{name}.en-ur.en"
    ur_path = paths.OPUS_UR_EN_DIR / f"{name}.en-ur.ur"
    hi_en_path = paths.OPUS_HI_EN_DIR / f"{name}.en-hi.en"
    hi_path = paths.OPUS_HI_EN_DIR / f"{name}.en-hi.hi"

    def build_pivot_map(en_path: Path, other_path: Path) -> dict[tuple[str, int], str]:
        # DAT-4: an earlier version dropped every English line that repeats
        # anywhere in the file, which biases the surviving sample toward
        # more distinctive, less-repeated sentences in an unquantified
        # direction. Disambiguating on (text, occurrence_index) instead
        # keeps duplicates -- the Nth occurrence of a repeated English line
        # on the ur-en side pivots against the Nth occurrence on the hi-en
        # side, still an exact match, just no longer requiring uniqueness.
        en_lines = en_path.read_text(encoding="utf-8").splitlines()
        other_lines = other_path.read_text(encoding="utf-8").splitlines()
        occurrence: dict[str, int] = {}
        mapping: dict[tuple[str, int], str] = {}
        for en_line, other_line in zip(en_lines, other_lines):
            en_key = en_line.strip()
            if not en_key:
                continue
            idx = occurrence.get(en_key, 0)
            occurrence[en_key] = idx + 1
            mapping[(en_key, idx)] = other_line
        return mapping

    ur_map = build_pivot_map(ur_en_path, ur_path)
    hi_map = build_pivot_map(hi_en_path, hi_path)
    shared = sorted(set(ur_map) & set(hi_map))

    # Diagnostic (DAT-4): report how many triples the old unique-only join
    # would have dropped (every key with occurrence_index > 0), and whether
    # the recovered duplicates skew shorter/longer than what already
    # survived the old scheme.
    if len(shared) and (n_dupe := sum(1 for k in shared if k[1] > 0)):
        dupe_len = sum(len(k[0]) for k in shared if k[1] > 0) / n_dupe
        unique_len = sum(len(k[0]) for k in shared if k[1] == 0) / max(1, len(shared) - n_dupe)
        print(f"[DAT-4] opus_{domain}: {n_dupe} triples recovered from repeated English pivot lines "
              f"(mean length {dupe_len:.1f} chars) vs {len(shared) - n_dupe} from unique lines (mean length {unique_len:.1f} chars)")

    if limit is not None:
        shared = shared[:limit]
    return [
        AlignedTriple(id=f"opus_{domain}_{i}", source=f"opus_{domain}", ur=ur_map[key], hi=hi_map[key], en=key[0])
        for i, key in enumerate(shared)
    ]


def _opus_pivot_map(en_path: Path, other_path: Path) -> dict[tuple[str, int], str]:
    # Same (text, occurrence_index) disambiguation as load_opus_pivot's
    # DAT-4 fix -- duplicated rather than shared to keep each loader
    # independently readable; not worth a shared helper for two call sites.
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    other_lines = other_path.read_text(encoding="utf-8").splitlines()
    occurrence: dict[str, int] = {}
    mapping: dict[tuple[str, int], str] = {}
    for en_line, other_line in zip(en_lines, other_lines):
        en_key = en_line.strip()
        if not en_key:
            continue
        idx = occurrence.get(en_key, 0)
        occurrence[en_key] = idx + 1
        mapping[(en_key, idx)] = other_line
    return mapping


def load_opus_ur_fa_pivot(domain: str = "globalvoices", limit: int | None = None) -> list[AlignedTriple]:
    """RQ2 follow-up EXP-7: Urdu/Persian triples pivoted on exact-string-
    matched English, same mechanism as load_opus_pivot (Urdu/Hindi) -- Persian
    shares Urdu's Perso-Arabic script but has no privileged relationship to
    Hindi/English, so this is the same-script-family, unrelated-competitor
    control RQ2's script-mass method needs an EXP-7 comparison for. Only
    GlobalVoices is available locally (see data/parallel/opus_fa_en/README)."""
    name = _OPUS_DOMAINS[domain.lower()]
    ur_en_path = paths.OPUS_UR_EN_DIR / f"{name}.en-ur.en"
    ur_path = paths.OPUS_UR_EN_DIR / f"{name}.en-ur.ur"
    fa_en_path = paths.OPUS_FA_EN_DIR / f"{name}.en-fa.en"
    fa_path = paths.OPUS_FA_EN_DIR / f"{name}.en-fa.fa"

    ur_map = _opus_pivot_map(ur_en_path, ur_path)
    fa_map = _opus_pivot_map(fa_en_path, fa_path)
    shared = sorted(set(ur_map) & set(fa_map))
    if limit is not None:
        shared = shared[:limit]
    return [
        AlignedTriple(id=f"opus_ur_fa_{domain}_{i}", source=f"opus_ur_fa_{domain}", ur=ur_map[key], fa=fa_map[key], meta={"en_pivot": key[0]})
        for i, key in enumerate(shared)
    ]


def load_roman_urdu_wordlist(top_n: int = 500) -> frozenset[str]:
    """Top-`top_n` most frequent Latin-alphabet words in the Roman-Urdu
    texting corpus (`data/translit/roman_urdu.jsonl`) -- used to disambiguate
    "Latin-script" from "English" (RQ2-3), since this file is the one corpus
    here that's genuinely Roman-Urdu texting register rather than literary
    transliteration or an English translation."""
    import re
    from collections import Counter

    counts: Counter[str] = Counter()
    path = paths.TRANSLIT_DIR / "roman_urdu.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("sentence", "")
            counts.update(re.findall(r"[a-zA-Z']+", text.lower()))
    return frozenset(word for word, _ in counts.most_common(top_n))


def load_roman_urdu_texting(limit: int | None = None) -> list[AlignedTriple]:
    """DAT-6: the one register everyday Urdu readers actually use (informal
    Roman-Urdu texting/social-media text) -- no parallel Hindi/English field,
    so this only supports single-language tasks (e.g. RQ2's logit lens), not
    RQ1's cross-language convergence."""
    path = paths.TRANSLIT_DIR / "roman_urdu.jsonl"
    triples: list[AlignedTriple] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("sentence", "")
            if not text:
                continue
            triples.append(AlignedTriple(id=f"roman_urdu_{i}", source="roman_urdu_texting", ur=text))
            if limit is not None and len(triples) >= limit:
                break
    return triples


LOADERS = {
    "rekhta_ghazals": lambda limit=None: load_rekhta("ghazals", limit=limit),
    "rekhta_nazms": lambda limit=None: load_rekhta("nazms", limit=limit),
    "flores_dev": lambda limit=None: load_flores("dev", limit=limit),
    "ebible": lambda limit=None: load_ebible(limit=limit),
}


def iter_all_sources(limit_per_source: int | None = None) -> Iterator[tuple[str, list[AlignedTriple]]]:
    for name, loader in LOADERS.items():
        yield name, loader(limit=limit_per_source)
