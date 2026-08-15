"""Hidden-state extraction and word/sentence pooling.

extract_hidden_states() is the only function here that touches a real model;
everything else is pure tensor/text bookkeeping and is unit-testable by
constructing a HiddenStates object directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch


@dataclass
class HiddenStates:
    layers: torch.Tensor  # (num_layers+1, seq_len, hidden_dim)
    offsets: list[tuple[int, int]]  # per-token (char_start, char_end), (0, 0) for special tokens
    text: str


def _altup_active_idx(model) -> int:
    """Gemma-3n's AltUp architecture (transformers' modeling_gemma3n.py)
    returns each layer's hidden state as [num_altup_inputs, batch, seq_len,
    hidden_size], not the plain [batch, seq_len, hidden_size] every other
    architecture here returns -- index config.altup_active_idx (default 0)
    is the "active" track the model itself reads out downstream (see
    Gemma3nTextAltUp.forward's use of predictions[altup_active_idx]), not an
    arbitrary choice. The gated multimodal top-level config doesn't carry
    this field directly; it lives on the nested text_config."""
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "altup_active_idx"):
            return cfg.altup_active_idx
    return 0


def extract_hidden_states(model, tokenizer, text: str, device: str = "cuda", max_length: int = 512) -> HiddenStates:
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden_states = out.hidden_states
    if hidden_states[0].dim() == 4:
        altup_idx = _altup_active_idx(model)
        hidden_states = tuple(hs[altup_idx] for hs in hidden_states)
    layers = torch.stack(hidden_states, dim=0)[:, 0].to(torch.float32).cpu()  # (L, T, H)
    return HiddenStates(layers=layers, offsets=offsets, text=text)


def word_char_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def word_span_vectors(hidden_states: HiddenStates) -> list[torch.Tensor | None]:
    """One (num_layers+1, hidden_dim) vector per whitespace-delimited word,
    mean-pooled over the subword tokens whose char span overlaps that word.
    None for a word with no overlapping real (non-special) token."""
    spans = word_char_spans(hidden_states.text)
    vectors: list[torch.Tensor | None] = []
    for start, end in spans:
        token_idxs = [
            i
            for i, (a, b) in enumerate(hidden_states.offsets)
            if b > a and a < end and b > start
        ]
        if not token_idxs:
            vectors.append(None)
            continue
        idx_t = torch.tensor(token_idxs)
        vectors.append(hidden_states.layers.index_select(1, idx_t).mean(dim=1))
    return vectors


def sentence_vector(hidden_states: HiddenStates) -> torch.Tensor:
    """(num_layers+1, hidden_dim) vector, mean-pooled over every real
    (non-special) token -- used for whole-sentence/verse comparisons where no
    word-level alignment is available."""
    real_idxs = [i for i, (a, b) in enumerate(hidden_states.offsets) if b > a]
    idx_t = torch.tensor(real_idxs)
    return hidden_states.layers.index_select(1, idx_t).mean(dim=1)
