import torch

from urdu_interp.extraction import HiddenStates, _altup_active_idx, sentence_vector, word_char_spans, word_span_vectors


class _Cfg:
    def __init__(self, altup_active_idx=None, text_config=None):
        if altup_active_idx is not None:
            self.altup_active_idx = altup_active_idx
        if text_config is not None:
            self.text_config = text_config


class _Model:
    def __init__(self, config):
        self.config = config


def test_altup_active_idx_defaults_to_zero_when_absent():
    assert _altup_active_idx(_Model(_Cfg())) == 0


def test_altup_active_idx_reads_top_level_config():
    assert _altup_active_idx(_Model(_Cfg(altup_active_idx=2))) == 2


def test_altup_active_idx_falls_back_to_nested_text_config():
    # Gemma-3n's gated multimodal top-level config doesn't carry this field
    # directly -- it lives on the nested text_config.
    model = _Model(_Cfg(text_config=_Cfg(altup_active_idx=3)))
    assert _altup_active_idx(model) == 3


def test_word_char_spans_splits_on_whitespace_and_newlines():
    text = "ab cd\nef"
    spans = word_char_spans(text)
    words = [text[a:b] for a, b in spans]
    assert words == ["ab", "cd", "ef"]


def _make_hidden_states() -> HiddenStates:
    text = "ab cd"
    # token 0 is a special token (offset (0,0)); "ab" is split into two
    # pieces ("a", "b"); "cd" is one piece.
    offsets = [(0, 0), (0, 1), (1, 2), (3, 5)]
    layer0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [10.0, 10.0, 10.0]])
    layer1 = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [20.0, 20.0, 20.0]])
    layers = torch.stack([layer0, layer1], dim=0)  # (2, 4, 3)
    return HiddenStates(layers=layers, offsets=offsets, text=text)


def test_word_span_vectors_pools_subwords_and_skips_special_tokens():
    hs = _make_hidden_states()
    vectors = word_span_vectors(hs)
    assert len(vectors) == 2
    assert torch.allclose(vectors[0], torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
    assert torch.allclose(vectors[1], torch.tensor([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]))


def test_word_span_vectors_none_when_no_overlapping_token():
    offsets = [(0, 0)]
    layers = torch.zeros(1, 1, 3)
    hs = HiddenStates(layers=layers, offsets=offsets, text="ab")
    vectors = word_span_vectors(hs)
    assert vectors == [None]


def test_sentence_vector_excludes_special_tokens():
    hs = _make_hidden_states()
    vec = sentence_vector(hs)
    expected_layer0 = (1.0 + 3.0 + 10.0) / 3
    expected_layer1 = (2.0 + 4.0 + 20.0) / 3
    assert torch.allclose(vec[0], torch.full((3,), expected_layer0))
    assert torch.allclose(vec[1], torch.full((3,), expected_layer1))
