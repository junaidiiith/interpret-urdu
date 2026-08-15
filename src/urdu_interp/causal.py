"""RQ2 follow-up EXP-4/EXP-5: causal intervention on the residual stream via
forward hooks on decoder-layer modules. Everything in logit_lens.py/
tuned_lens.py only *reads* a representation; nothing in this project so far
produces causal evidence that a decodable representation actually
participates in the model's real output, rather than being present but
functionally inert. EXP-4 (swap in a borrowed hidden state) and EXP-5 (add a
steering vector) share this hook mechanism.

Layer indexing matches HiddenStates.layers (extraction.py) throughout this
project: layer 0 is the embedding output, layer l>=1 is decoder block l-1's
output. Every intervention here targets layer >= 1 (patching the raw
embedding layer, before any decoder block runs, isn't supported).

Scope note: only tested against plain decoder-only layouts (Qwen3.5,
SmolLM3), confirmed here to return a plain tensor (not a tuple) from each
decoder layer's forward. Gemma-3n's AltUp architecture is deliberately out of
scope for this module -- its 4-track hidden states (extraction.py's
_altup_active_idx handling) mean a naive (batch, seq, hidden) overwrite hook
is not verified safe there, and getting it right needs the same kind of
architecture-specific check ENG-2 already had to do for extraction.py/
logit_lens.py. Not attempted here rather than risking a silent bug.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


def resolve_decoder_layers(model):
    """Same layout-search pattern as logit_lens._resolve_final_norm."""
    candidate_paths = [
        (model, "model", "layers"),
        (model, "layers"),
        (model, "language_model", "layers"),
        (model, "model", "language_model", "layers"),
    ]
    for path in candidate_paths:
        obj = path[0]
        for attr in path[1:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            if obj is not None:
                return obj
    raise AttributeError("could not locate a decoder-layer list on this model")


def _unwrap(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


@contextmanager
def patch_hidden_state(model, layer: int, position: int, replacement: torch.Tensor) -> Iterator[None]:
    """Overwrite decoder layer `layer`'s output hidden state at `position`
    with `replacement` (shape (hidden_dim,)) for every forward call made
    inside this context -- EXP-4's cross-lingual swap."""
    layers = resolve_decoder_layers(model)

    def hook(module, args, output):
        hidden = _unwrap(output).clone()
        hidden[:, position, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
        return _rewrap(output, hidden)

    handle = layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def steer_hidden_state(model, layer: int, direction: torch.Tensor, alpha: float) -> Iterator[None]:
    """Add `alpha * direction` (shape (hidden_dim,)) to decoder layer
    `layer`'s output hidden state at every position, for every forward call
    inside this context -- applies continuously through generation, not just
    the prompt, which is EXP-5's dose-response design."""
    layers = resolve_decoder_layers(model)

    def hook(module, args, output):
        hidden = _unwrap(output).clone()
        delta = (alpha * direction).to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + delta
        return _rewrap(output, hidden)

    handle = layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
