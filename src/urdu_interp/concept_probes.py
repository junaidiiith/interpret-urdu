"""RQ2 follow-up EXP-3: the free-text next-word task (tasks.py/logit_lens.py)
buckets every top-k token by *script*, which mixes function words,
punctuation, and content words into one number -- there's no per-item ground
truth. This gives each item a single unambiguous correct answer instead: a
word-repetition task (a few fixed Urdu word->word exemplars establishing
"echo the input back exactly", then a target Urdu word to repeat) borrowed
from Wendler et al. 2024's "Do Llamas Work in English?" design. The
concept's Hindi and English translations are never shown to the model --
they're only used to ask "at each layer, does the logit lens favor the
correct (Urdu, repeat-task) token, or does it look like the model is about
to say the semantically equivalent Hindi/English word instead?"

Caveat, disclosed rather than glossed over: many of these words are more
than one subword token in a given model's tokenizer. Only the first token is
used as the scored target -- an approximation, not a guarantee that "the
model said this token" means "the model said this whole word."
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from urdu_interp.logit_lens import UnembedFn, _resolve_unembed
from urdu_interp.stats import first_stable_index

_FEWSHOT_UR = ["پتھر", "کاغذ"]  # stone, paper -- fixed exemplars, never scored
_TEMPLATE = "لفظ: {a} → {a}\nلفظ: {b} → {b}\nلفظ: {target} → "


def build_prompt(target_ur: str) -> str:
    a, b = _FEWSHOT_UR
    return _TEMPLATE.format(a=a, b=b, target=target_ur)


def first_token_id(tokenizer, word: str) -> int:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for {word!r}")
    return ids[0]


@dataclass
class ConceptProbeResult:
    concept_id: str
    layer_probs: list[dict[str, float]]  # per layer: {"ur": p, "hi": p, "en": p}
    resolved_correctly_final_layer: bool
    crossover_layer: int | None  # earliest layer from which ur > hi and ur > en for the rest of the network


def run_concept_probe(
    hidden_states,
    unembed_fn: UnembedFn | list[UnembedFn],
    ur_token_id: int,
    hi_token_id: int,
    en_token_id: int,
    position: int = -1,
) -> list[dict[str, float]]:
    num_layers = hidden_states.layers.shape[0]
    rows = []
    for layer in range(num_layers):
        logits = _resolve_unembed(unembed_fn, layer)(hidden_states.layers[layer, position, :])
        probs = torch.softmax(logits.float(), dim=-1)
        rows.append({"ur": probs[ur_token_id].item(), "hi": probs[hi_token_id].item(), "en": probs[en_token_id].item()})
    return rows


def concept_crossover_layer(layer_probs: list[dict[str, float]]) -> int | None:
    flags = [row["ur"] > row["hi"] and row["ur"] > row["en"] for row in layer_probs]
    return first_stable_index(flags)
