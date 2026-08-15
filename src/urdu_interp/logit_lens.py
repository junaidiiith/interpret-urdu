"""RQ2: the 'logit lens'. We apply the model's own final read-out (its last
normalization + unembedding matrix) to an *intermediate* layer's hidden
state, even though the model was never trained to be read out there -- a
forced, artificial 'what would you say right now' probe, not something the
model does on its own. Only make_unembed_fn touches real model internals;
everything downstream is testable with a fake unembed_fn.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from urdu_interp.extraction import HiddenStates
from urdu_interp.script_utils import SCRIPTS, classify_token
from urdu_interp.stats import first_stable_index

UnembedFn = Callable[[torch.Tensor], torch.Tensor]
# Must decode a token id to its actual text (e.g. tokenizer.decode([id])),
# NOT tokenizer.convert_ids_to_tokens -- for byte-level BPE tokenizers the
# latter returns the raw byte-remapped vocab string (Urdu 'مح' round-trips
# as the unrelated-looking 'ÙħØŃ'), which silently breaks script
# classification instead of raising.
TokenLookupFn = Callable[[int], str]


def _resolve_final_norm(model):
    """Tries the plain decoder-only layout first (model.norm or
    model.model.norm), then the layouts used by multimodal-capable
    checkpoints, which nest the text decoder under a language_model
    sub-module -- one level deep (model.language_model.norm) or two
    (model.model.language_model.norm, confirmed for Gemma-3n: its top-level
    Gemma3nForConditionalGeneration.model is a Gemma3nModel wrapping the
    actual Gemma3nTextModel as .language_model, which is what holds .norm).
    Returns None if none of these paths resolve."""
    candidate_paths = [
        (model,),
        (model, "model"),
        (model, "language_model"),
        (model, "model", "language_model"),
    ]
    for path in candidate_paths:
        obj = path[0]
        for attr in path[1:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "norm"):
            return obj.norm
    return None


def make_unembed_fn(model) -> UnembedFn:
    """Resolves the final norm + output-embedding (lm_head) modules for a HF
    causal LM (see _resolve_final_norm for the layouts tried). Raises if
    neither is found -- callers can always build and pass their own
    UnembedFn instead of relying on this."""
    norm = _resolve_final_norm(model)
    if norm is None:
        raise AttributeError("could not locate a final-norm module on this model; pass an explicit UnembedFn")

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("model.get_output_embeddings() returned None; pass an explicit UnembedFn")

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def unembed(hidden: torch.Tensor) -> torch.Tensor:
        # extraction.py stores hidden states on CPU in float32; move back to
        # the model's device/dtype before running its own norm + lm_head.
        with torch.no_grad():
            hidden = hidden.to(device=model_device, dtype=model_dtype)
            return lm_head(norm(hidden)).float()

    return unembed


LatinSubtypeFn = Callable[[str], str]


def _resolve_unembed(unembed_fn: UnembedFn | list[UnembedFn], layer: int) -> UnembedFn:
    # RQ2-1: a tuned lens needs a different affine map per layer, unlike the
    # raw lens's single shared unembedding -- accept either a single callable
    # (raw lens, existing behavior) or a list indexed by layer (tuned lens).
    if isinstance(unembed_fn, list):
        return unembed_fn[layer]
    return unembed_fn


def layer_script_mass(
    logits: torch.Tensor,
    convert_id_to_token: TokenLookupFn,
    top_k: int = 50,
    latin_subtype_fn: LatinSubtypeFn | None = None,
) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    k = min(top_k, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k)
    mass = dict.fromkeys(SCRIPTS, 0.0)
    if latin_subtype_fn is not None:
        mass["latin_english"] = 0.0
        mass["latin_roman_urdu"] = 0.0
        mass["latin_ambiguous"] = 0.0
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        token = convert_id_to_token(idx)
        script = classify_token(token)
        mass[script] += p
        # RQ2-3: "Latin script" is a Unicode-block fact, not a language --
        # break the latin bucket down further so "English" isn't asserted
        # without evidence beyond script membership.
        if script == "latin" and latin_subtype_fn is not None:
            mass[f"latin_{latin_subtype_fn(token)}"] += p
    return mass


def run_layer_script_masses(
    hidden_states: HiddenStates,
    unembed_fn: UnembedFn | list[UnembedFn],
    convert_id_to_token: TokenLookupFn,
    position: int = -1,
    top_k: int = 50,
    latin_subtype_fn: LatinSubtypeFn | None = None,
) -> list[dict[str, float]]:
    """Script-mass breakdown at every layer for one token position (default:
    the last position, i.e. what the model currently believes the next token
    is)."""
    num_layers = hidden_states.layers.shape[0]
    return [
        layer_script_mass(
            _resolve_unembed(unembed_fn, layer)(hidden_states.layers[layer, position, :]),
            convert_id_to_token,
            top_k=top_k,
            latin_subtype_fn=latin_subtype_fn,
        )
        for layer in range(num_layers)
    ]


@dataclass
class DetourResult:
    layer_masses: list[dict[str, float]]
    native_crossover_layer: int | None


def native_crossover_layer(layer_masses: list[dict[str, float]], target: str, competitors: list[str]) -> int | None:
    """Earliest layer from which the target script's mass exceeds every
    competitor script's mass for the rest of the network -- the point at
    which the model stops 'leaning towards' a competitor language and
    commits to the target. None if it never stabilizes that way."""
    flags = [m[target] > max(m[c] for c in competitors) for m in layer_masses]
    return first_stable_index(flags)


def detect_detour(
    hidden_states: HiddenStates,
    unembed_fn: UnembedFn | list[UnembedFn],
    convert_id_to_token: TokenLookupFn,
    target: str,
    competitors: list[str],
    position: int = -1,
    top_k: int = 50,
    latin_subtype_fn: LatinSubtypeFn | None = None,
) -> DetourResult:
    masses = run_layer_script_masses(
        hidden_states, unembed_fn, convert_id_to_token, position=position, top_k=top_k, latin_subtype_fn=latin_subtype_fn
    )
    return DetourResult(layer_masses=masses, native_crossover_layer=native_crossover_layer(masses, target, competitors))
