import json

from urdu_interp import corpora, paths


def test_load_rekhta_reads_ur_hi_en_rm_fields(tmp_path, monkeypatch):
    poetry_dir = tmp_path / "poetry"
    poetry_dir.mkdir()
    data = {
        "https://rekhta/poets/x": {
            "https://rekhta/ghazals/1": {
                "ur": "محبت تو مت کر",
                "hi": "मोहब्बत तू मत कर",
                "en": "mohabbat tū mat kar",
                "en-rm": "mohabbat tu mat kar",
            },
            "https://rekhta/ghazals/2": {
                "ur": "incomplete entry, missing hi/en-rm",
            },
        },
        "https://rekhta/poets/y": {},
    }
    (poetry_dir / "rekhta_all_poets_ghazals.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(paths, "POETRY_DIR", poetry_dir)

    triples = corpora.load_rekhta("ghazals")

    assert len(triples) == 1
    t = triples[0]
    assert t.ur == "محبت تو مت کر"
    assert t.hi == "मोहब्बत तू मत कर"
    assert t.en == "mohabbat tu mat kar"
    assert t.source == "rekhta_ghazals"
    assert t.meta["poet"] == "https://rekhta/poets/x"


def test_load_rekhta_respects_limit(tmp_path, monkeypatch):
    poetry_dir = tmp_path / "poetry"
    poetry_dir.mkdir()
    poems = {
        f"https://rekhta/ghazals/{i}": {"ur": "a b", "hi": "c d", "en-rm": "e f"} for i in range(5)
    }
    data = {"https://rekhta/poets/x": poems}
    (poetry_dir / "rekhta_all_poets_ghazals.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(paths, "POETRY_DIR", poetry_dir)

    triples = corpora.load_rekhta("ghazals", limit=2)
    assert len(triples) == 2


def test_load_flores_zips_by_line_number(tmp_path, monkeypatch):
    flores_dir = tmp_path / "flores"
    dev_dir = flores_dir / "dev"
    dev_dir.mkdir(parents=True)
    (dev_dir / "urd_Arab.dev").write_text("ur1\nur2\n", encoding="utf-8")
    (dev_dir / "hin_Deva.dev").write_text("hi1\nhi2\n", encoding="utf-8")
    (dev_dir / "eng_Latn.dev").write_text("en1\nen2\n", encoding="utf-8")
    monkeypatch.setattr(paths, "FLORES_DIR", flores_dir)

    triples = corpora.load_flores("dev")
    assert len(triples) == 2
    assert triples[0].ur == "ur1"
    assert triples[0].hi == "hi1"
    assert triples[0].en == "en1"
    assert triples[1].ur == "ur2"


def test_load_flores_truncates_to_shortest_file(tmp_path, monkeypatch):
    flores_dir = tmp_path / "flores"
    dev_dir = flores_dir / "dev"
    dev_dir.mkdir(parents=True)
    (dev_dir / "urd_Arab.dev").write_text("ur1\nur2\nur3\n", encoding="utf-8")
    (dev_dir / "hin_Deva.dev").write_text("hi1\n", encoding="utf-8")
    (dev_dir / "eng_Latn.dev").write_text("en1\nen2\n", encoding="utf-8")
    monkeypatch.setattr(paths, "FLORES_DIR", flores_dir)

    triples = corpora.load_flores("dev")
    assert len(triples) == 1


def test_load_ebible_joins_on_shared_verse_key(tmp_path, monkeypatch):
    ebible_dir = tmp_path / "ebible"
    ebible_dir.mkdir()
    (ebible_dir / "urdgvu_vpl.txt").write_text("GEN 1:1 ur-one\nGEN 1:2 ur-two\n", encoding="utf-8")
    (ebible_dir / "hincv_vpl.txt").write_text("GEN 1:1 hi-one\n", encoding="utf-8")  # missing 1:2
    (ebible_dir / "eng-web_vpl.txt").write_text("GEN 1:1 en-one\nGEN 1:2 en-two\n", encoding="utf-8")
    monkeypatch.setattr(paths, "EBIBLE_DIR", ebible_dir)

    triples = corpora.load_ebible("urdgvu", "hincv", "eng-web")
    assert len(triples) == 1
    assert triples[0].id == "GEN 1:1"
    assert triples[0].ur == "ur-one"
    assert triples[0].hi == "hi-one"
    assert triples[0].en == "en-one"


def test_load_opus_pivot_joins_on_exact_english_match_disambiguated_by_occurrence(tmp_path, monkeypatch):
    ur_en_dir = tmp_path / "opus_ur_en"
    hi_en_dir = tmp_path / "opus_hi_en"
    ur_en_dir.mkdir()
    hi_en_dir.mkdir()
    # "dup" appears twice on the ur-en English side but once on the hi-en
    # side -- DAT-4: disambiguate by (text, occurrence_index) instead of
    # dropping the whole key, so the first "dup" occurrence still pivots
    # (only the second, with no hi-side counterpart, is left out).
    (ur_en_dir / "TED2020.en-ur.en").write_text("hello\ndup\ndup\nonly-ur\n", encoding="utf-8")
    (ur_en_dir / "TED2020.en-ur.ur").write_text("ur-hello\nur-dup-1\nur-dup-2\nur-only\n", encoding="utf-8")
    (hi_en_dir / "TED2020.en-hi.en").write_text("hello\ndup\nonly-hi\n", encoding="utf-8")
    (hi_en_dir / "TED2020.en-hi.hi").write_text("hi-hello\nhi-dup\nhi-only\n", encoding="utf-8")
    monkeypatch.setattr(paths, "OPUS_UR_EN_DIR", ur_en_dir)
    monkeypatch.setattr(paths, "OPUS_HI_EN_DIR", hi_en_dir)

    triples = corpora.load_opus_pivot("ted2020")
    by_en = {t.en: t for t in triples}
    assert len(triples) == 2
    assert by_en["hello"].ur == "ur-hello"
    assert by_en["hello"].hi == "hi-hello"
    assert by_en["dup"].ur == "ur-dup-1"
    assert by_en["dup"].hi == "hi-dup"


def test_load_roman_urdu_texting_reads_sentence_field(tmp_path, monkeypatch):
    translit_dir = tmp_path / "translit"
    translit_dir.mkdir()
    (translit_dir / "roman_urdu.jsonl").write_text(
        '{"sentence": "kya baat hai", "sentiment": 0}\n'
        '{"sentence": "", "sentiment": 0}\n'
        '{"sentence": "acha theek hai", "sentiment": 1}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "TRANSLIT_DIR", translit_dir)

    triples = corpora.load_roman_urdu_texting()
    assert len(triples) == 2  # blank sentence skipped
    assert triples[0].ur == "kya baat hai"
    assert triples[0].hi is None
    assert triples[0].source == "roman_urdu_texting"


def test_load_roman_urdu_texting_respects_limit(tmp_path, monkeypatch):
    translit_dir = tmp_path / "translit"
    translit_dir.mkdir()
    (translit_dir / "roman_urdu.jsonl").write_text(
        "\n".join(f'{{"sentence": "line {i}", "sentiment": 0}}' for i in range(5)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "TRANSLIT_DIR", translit_dir)

    triples = corpora.load_roman_urdu_texting(limit=2)
    assert len(triples) == 2


def test_load_roman_urdu_wordlist_ranks_by_frequency(tmp_path, monkeypatch):
    translit_dir = tmp_path / "translit"
    translit_dir.mkdir()
    (translit_dir / "roman_urdu.jsonl").write_text(
        '{"sentence": "kya kya kya hai hai acha"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "TRANSLIT_DIR", translit_dir)

    wordlist = corpora.load_roman_urdu_wordlist(top_n=2)
    assert wordlist == frozenset({"kya", "hai"})
