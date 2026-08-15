"""RQ2 follow-up EXP-8: TCAV-style concept attribution (Kim et al. 2018,
"Interpretability Beyond Feature Attribution: Testing with Concept
Activation Vectors"), adapted to sit on top of EXP-4/EXP-5's causal
intervention machinery rather than TCAV's original gradient-based
directional-derivative formulation.

What this adds that EXP-4/EXP-5 didn't have: neither established that the
causal effect of pushing the residual stream in an "English" direction is
*specific* to that direction, as opposed to any perturbation of similar
magnitude producing a similar shift -- there was no negative control. This
module fits a proper linear-classifier concept activation vector (CAV, not
just a difference-of-means direction) for "English vs. Urdu" at a layer,
then builds a null distribution of CAVs by refitting the identical
classifier on randomly permuted labels over the same activations. Testing
the real CAV's causal effect (via causal.steer_hidden_state, the same
mechanism EXP-5 uses) against that null distribution gives an actual
permutation-test p-value: is the real concept direction's effect bigger
than what a same-shaped-but-meaningless direction would produce?

Scope: inherits causal.py's Qwen3.5/SmolLM3-only restriction (Gemma-3n's
AltUp architecture is not verified safe for the underlying hook).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from urdu_interp import causal, extraction, logit_lens


def sentence_vectors_at_layer(model, tokenizer, texts: list[str], layer: int, device: str) -> torch.Tensor:
    vecs = []
    for t in texts:
        hs = extraction.extract_hidden_states(model, tokenizer, t, device=device)
        vecs.append(extraction.sentence_vector(hs)[layer])
    return torch.stack(vecs)


def fit_cav(pos_vecs: torch.Tensor, neg_vecs: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """A CAV is the (normalized) weight vector of a linear classifier
    trained to separate two labeled activation sets -- here English-context
    (positive) vs. Urdu-context (negative) sentence vectors at one layer.
    This is the actual TCAV construction, not a difference-of-means
    shortcut: the classifier can weight dimensions unevenly, and L2
    regularization keeps it well-posed even though hidden_dim can exceed
    the number of examples."""
    X = torch.cat([pos_vecs, neg_vecs], dim=0).numpy()
    y = np.array([1] * pos_vecs.shape[0] + [0] * neg_vecs.shape[0])
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(X, y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return w / w.norm()


def fit_null_cavs(pos_vecs: torch.Tensor, neg_vecs: torch.Tensor, n_null: int, seed: int = 0) -> list[torch.Tensor]:
    """Null distribution: refit the identical classifier on the identical
    pooled activations, but with the English/Urdu labels randomly permuted
    each time. Each null CAV is "a direction that happens to separate some
    arbitrary random split of this same data" -- same fitting procedure,
    same activations, no real concept behind the split."""
    X = torch.cat([pos_vecs, neg_vecs], dim=0).numpy()
    original_labels = np.array([1] * pos_vecs.shape[0] + [0] * neg_vecs.shape[0])
    rng = np.random.default_rng(seed)
    null_cavs = []
    for i in range(n_null):
        # keep X in its original row order; randomly reassign which row
        # gets which label -- the standard permutation-test null ("does
        # this label assignment carry real signal, or would any random
        # relabeling of the same activations do just as well").
        shuffled_labels = rng.permutation(original_labels)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed + i + 1)
        clf.fit(X, shuffled_labels)
        w = torch.tensor(clf.coef_[0], dtype=torch.float32)
        null_cavs.append(w / w.norm())
    return null_cavs


@dataclass
class CavEffect:
    mean_delta_urdu: float
    mean_delta_latin: float
    mean_delta_hindi: float
    per_item_delta_latin: list[float]


def measure_cav_effect(
    model, tokenizer, direction: torch.Tensor, prefixes: list[str], layer: int, alpha: float, device: str,
    top_k: int = 50, latin_subtype_fn=None,
) -> CavEffect:
    """Single-forward-pass causal measurement -- add alpha*direction to the
    residual stream at `layer` for one forward call on each Urdu prefix,
    then compare the model's REAL final-layer logits (not a lens) against
    an unsteered baseline. Deliberately the same statistic family as
    EXP-4's activation patching (Delta script-mass on real output logits),
    not EXP-5's full-generation code-switch rate -- cheaper, cleaner, and
    lets many null CAVs be tested at the scale EXP-4 already validated."""
    def decode_id(tid: int) -> str:
        return tokenizer.decode([tid])

    deltas_urdu, deltas_latin, deltas_hindi = [], [], []
    for prefix in prefixes:
        enc = tokenizer(prefix, return_tensors="pt").to(device)
        with torch.no_grad():
            baseline_logits = model(**enc).logits[0, -1, :].float().cpu()
        with causal.steer_hidden_state(model, layer=layer, direction=direction, alpha=alpha):
            with torch.no_grad():
                steered_logits = model(**enc).logits[0, -1, :].float().cpu()

        baseline_mass = logit_lens.layer_script_mass(baseline_logits, decode_id, top_k=top_k, latin_subtype_fn=latin_subtype_fn)
        steered_mass = logit_lens.layer_script_mass(steered_logits, decode_id, top_k=top_k, latin_subtype_fn=latin_subtype_fn)
        deltas_urdu.append(steered_mass["urdu"] - baseline_mass["urdu"])
        deltas_latin.append(steered_mass["latin"] - baseline_mass["latin"])
        deltas_hindi.append(steered_mass["hindi"] - baseline_mass["hindi"])

    return CavEffect(
        mean_delta_urdu=sum(deltas_urdu) / len(deltas_urdu),
        mean_delta_latin=sum(deltas_latin) / len(deltas_latin),
        mean_delta_hindi=sum(deltas_hindi) / len(deltas_hindi),
        per_item_delta_latin=deltas_latin,
    )


def permutation_test(real_effect: float, null_effects: list[float]) -> dict:
    """Standard permutation-test p-value with the +1/+1 continuity
    correction (so p is never reported as exactly 0 off a finite null
    sample), plus a z-score against the null distribution's mean/std."""
    null_arr = np.array(null_effects)
    n_null = len(null_effects)
    n_as_extreme = int(np.sum(null_arr >= real_effect))
    p_value = (1 + n_as_extreme) / (1 + n_null)
    mean, std = float(null_arr.mean()), float(null_arr.std())
    z_score = (real_effect - mean) / std if std > 0 else float("inf")
    return {
        "real_effect": real_effect,
        "null_mean": mean,
        "null_std": std,
        "z_score": z_score,
        "p_value": p_value,
        "n_null": n_null,
    }
