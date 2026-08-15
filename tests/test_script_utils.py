from urdu_interp.script_utils import classify_char, classify_latin_subtype, classify_token


def test_classify_char_urdu():
    assert classify_char("ا") == "urdu"  # ALEF, U+0627


def test_classify_char_hindi():
    assert classify_char("क") == "hindi"  # KA, U+0915


def test_classify_char_latin():
    assert classify_char("a") == "latin"


def test_classify_char_other_for_digits_and_punctuation():
    assert classify_char("5") == "other"
    assert classify_char(".") == "other"


def test_classify_token_urdu_word():
    assert classify_token("محبت") == "urdu"


def test_classify_token_hindi_word():
    assert classify_token("मोहब्बत") == "hindi"


def test_classify_token_latin_word():
    assert classify_token("love") == "latin"


def test_classify_token_ignores_subword_marker_characters():
    # sentencepiece marker U+2581 is in none of the tracked ranges
    assert classify_token("▁love") == "latin"


def test_classify_token_majority_vote_on_mixed_token():
    assert classify_token("abمحبت") == "urdu"  # 4 urdu chars vs 2 latin chars


def test_classify_token_other_when_no_script_characters():
    assert classify_token("123!!") == "other"


_ROMAN_URDU_WORDS = frozenset({"kya", "hai", "nahi", "acha"})
_ENGLISH_WORDS = frozenset({"what", "is", "not", "good"})


def test_classify_latin_subtype_english_word():
    assert classify_latin_subtype("What", _ROMAN_URDU_WORDS, _ENGLISH_WORDS) == "english"


def test_classify_latin_subtype_roman_urdu_word():
    assert classify_latin_subtype("Kya", _ROMAN_URDU_WORDS, _ENGLISH_WORDS) == "roman_urdu"


def test_classify_latin_subtype_ambiguous_when_in_neither_list():
    assert classify_latin_subtype("xyz", _ROMAN_URDU_WORDS, _ENGLISH_WORDS) == "ambiguous"


def test_classify_latin_subtype_ambiguous_when_in_both_lists():
    both = frozenset({"so"})
    assert classify_latin_subtype("so", both, both) == "ambiguous"


def test_classify_latin_subtype_empty_token_is_ambiguous():
    assert classify_latin_subtype("###", _ROMAN_URDU_WORDS, _ENGLISH_WORDS) == "ambiguous"


def test_classify_latin_subtype_uses_default_english_wordlist():
    assert classify_latin_subtype("the", frozenset()) == "english"
