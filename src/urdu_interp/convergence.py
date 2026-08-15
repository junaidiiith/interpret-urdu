"""RQ1: does a model's internal representation of matched Urdu/Hindi (or
Urdu/English) content converge layer by layer, and if so, how early?

All functions here operate on already-extracted (N, num_layers, hidden_dim)
tensors, matched pairwise by index -- they don't know or care whether the
vectors came from a real model or a test fixture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from urdu_interp.stats import first_stable_index


def center(vectors: torch.Tensor) -> torch.Tensor:
    """Subtract the column mean -- removes the generic 'which language is
    this' direction shared by every vector in the batch, so cosine similarity
    reflects content rather than a constant per-language offset."""
    return vectors - vectors.mean(dim=0, keepdim=True)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def centered_cosine_per_layer(vecs_a: torch.Tensor, vecs_b: torch.Tensor) -> torch.Tensor:
    """(N, L) matrix of per-item, per-layer cosine similarity after centering
    each layer's vectors independently."""
    n, num_layers, _ = vecs_a.shape
    out = torch.zeros(n, num_layers)
    for layer in range(num_layers):
        a = center(vecs_a[:, layer, :])
        b = center(vecs_b[:, layer, :])
        out[:, layer] = cosine_sim(a, b)
    return out


def union_layer_mean(batches: list[tuple[torch.Tensor, list[str]]]) -> torch.Tensor:
    """Per-layer mean over the *deduplicated union* of word vectors across
    multiple batches of the same language (e.g. Urdu vectors extracted once
    for the ur-hi run and once for the ur-en run) -- the same
    (triple_id, word_index) pair gets the same contextual vector regardless
    of which run extracted it (the extraction only ever depends on that
    language's own text), so a plain dict keyed by id dedupes correctly.
    Returns (num_layers, hidden_dim). Used so a language's centering mean
    (RQ1-3) doesn't silently shift depending on which other language it's
    being compared against."""
    seen: dict[str, torch.Tensor] = {}
    for vecs, ids in batches:
        for vec, id_ in zip(vecs, ids):
            seen[id_] = vec
    if not seen:
        raise ValueError("no vectors to compute a union mean from")
    return torch.stack(list(seen.values())).mean(dim=0)


@dataclass
class ConvergenceResult:
    n_pairs: int
    layer_scores: list[float]  # mean real cosine similarity per layer, centered
    null_mean: list[float]
    null_std: list[float]
    effect_size: list[float]  # (real - null_mean) / null_std, per layer
    per_item_scores: torch.Tensor  # (N, L) real (non-shuffled) per-item scores


def layerwise_convergence(
    vecs_a: torch.Tensor,
    vecs_b: torch.Tensor,
    n_null: int = 20,
    seed: int = 0,
    mean_a: torch.Tensor | None = None,
    mean_b: torch.Tensor | None = None,
) -> ConvergenceResult:
    """Per-layer convergence between two sets of matched vectors, reported as
    an effect size against a shuffled-pair null (pairing each item's `a`
    vector with a random *other* item's `b` vector) -- the same selectivity
    discipline used in probing-classifier control tasks, so the number
    reported isn't just "how similar are two centered random vectors on
    average" but "how much more similar are the *true* pairs than that".

    `mean_a`/`mean_b` (each (num_layers, hidden_dim), optional): center
    against an externally-supplied per-layer mean instead of this batch's
    own mean -- e.g. `union_layer_mean` computed over a wider reference
    population (RQ1-3), so the same language centers the same way across
    different lang-pair comparisons. None (default) reproduces the original
    per-batch-mean behavior exactly."""
    n, num_layers, _ = vecs_a.shape

    def centered(vecs: torch.Tensor, layer: int, mean: torch.Tensor | None) -> torch.Tensor:
        m = mean[layer] if mean is not None else vecs[:, layer, :].mean(dim=0, keepdim=True)
        return vecs[:, layer, :] - m

    real_per_item = torch.zeros(n, num_layers)
    for layer in range(num_layers):
        real_per_item[:, layer] = cosine_sim(centered(vecs_a, layer, mean_a), centered(vecs_b, layer, mean_b))
    layer_scores = real_per_item.mean(dim=0).tolist()

    rng = np.random.default_rng(seed)
    null_means: list[float] = []
    null_stds: list[float] = []
    for layer in range(num_layers):
        a = centered(vecs_a, layer, mean_a)
        b = centered(vecs_b, layer, mean_b)
        # Pool individual shuffled-pair scores across shuffles (n_null * N
        # values), not the n_null shuffle-level means -- the null we want is
        # "how similar is one item to a random other item", not "how much
        # does the *average* shuffled score vary between repeated shuffles"
        # (the latter shrinks with N and inflates the effect size).
        null_scores = []
        for _ in range(n_null):
            perm = rng.permutation(n)
            null_scores.extend(cosine_sim(a, b[perm]).tolist())
        null_means.append(float(np.mean(null_scores)))
        null_stds.append(float(np.std(null_scores)) or 1e-8)

    effect_size = [
        (layer_scores[layer] - null_means[layer]) / null_stds[layer] for layer in range(num_layers)
    ]
    return ConvergenceResult(
        n_pairs=n,
        layer_scores=layer_scores,
        null_mean=null_means,
        null_std=null_stds,
        effect_size=effect_size,
        per_item_scores=real_per_item,
    )


def recovery_layer(effect_size: list[float], threshold: float = 0.5) -> int | None:
    """Earliest layer from which the effect size stays at/above `threshold`
    for the rest of the network. None if it never stabilizes there -- read
    as "the gap never resolves", the deep-capacity-gap outcome for RQ1.

    Secondary diagnostic only: on this project's real runs the effect size
    clears `threshold` at layer 0-1 for essentially every model/corpus/
    lang-pair combination, so this binary flag doesn't discriminate
    anything -- see `peak_layer_frac`/`effect_size_auc` for the primary
    reported statistics."""
    flags = [e >= threshold for e in effect_size]
    return first_stable_index(flags)


def peak_layer_frac(layer_scores: list[float]) -> float:
    """Depth-normalized location of the raw peak score (0 = first layer, 1 =
    last layer). This is where the real texture of RQ1 lives -- e.g. eBible
    peaking early vs. Rekhta peaking late -- not captured by the saturated
    binary `recovery_layer` flag."""
    n = len(layer_scores)
    if n <= 1:
        return 0.0
    peak_idx = max(range(n), key=lambda i: layer_scores[i])
    return peak_idx / (n - 1)


def effect_size_auc(effect_size: list[float]) -> float:
    """Mean effect size across all layers (a depth-normalized area under the
    curve) -- a single-number summary of how strongly convergence holds
    across the whole network, not just at one threshold-crossing layer."""
    if not effect_size:
        return 0.0
    return sum(effect_size) / len(effect_size)
