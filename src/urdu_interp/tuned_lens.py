"""RQ2-1: a minimal tuned lens. The raw (untuned) logit lens applies the
model's frozen final norm + unembedding directly to early/middle-layer
hidden states, which is known to be unreliable there -- the model was never
trained to be read out at that point. The standard fix (Belrose et al.'s
"tuned lens") is a small per-layer affine probe trained to translate that
layer's hidden state into the final layer's representation space before
applying the frozen readout. This implements that as closed-form ridge
regression per layer (fit on a modest slice of ordinary text) rather than
pulling in an external tuned-lens library or doing gradient-descent training.
"""
from __future__ import annotations

import torch

from urdu_interp import extraction
from urdu_interp.logit_lens import UnembedFn, make_unembed_fn

LensParams = list[tuple[torch.Tensor, torch.Tensor]]  # one (W, b) per layer


def fit_tuned_lens(model, tokenizer, texts: list[str], device: str, ridge: float = 1e-2, max_length: int = 128) -> LensParams:
    """Fit one affine map per layer: predicted_final_hidden = h_l @ W + b,
    least-squares (ridge-regularized, closed form) over every real token
    position across `texts`, targeting that same token position's
    final-layer hidden state. Returns one (W, b) per layer, indexed the same
    way as HiddenStates.layers (embedding layer first)."""
    per_text_layers = [extraction.extract_hidden_states(model, tokenizer, t, device=device, max_length=max_length).layers for t in texts]
    num_layers = per_text_layers[0].shape[0]
    hidden_dim = per_text_layers[0].shape[-1]

    # (num_layers, total_tokens, hidden_dim) across every text concatenated
    stacked = torch.cat([layers.reshape(num_layers, -1, hidden_dim) for layers in per_text_layers], dim=1)
    target = stacked[-1]  # final layer's hidden states are the regression target
    n_tokens = target.shape[0]
    ones = torch.ones(n_tokens, 1)

    lens_params: LensParams = []
    for layer in range(num_layers):
        design = torch.cat([stacked[layer], ones], dim=1)  # (T, H+1), affine via augmented design matrix
        gram = design.T @ design
        solution = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0]), design.T @ target)
        lens_params.append((solution[:-1, :], solution[-1, :]))  # (W: H x H, b: H)
    return lens_params


def make_tuned_unembed_fns(model, lens_params: LensParams) -> list[UnembedFn]:
    """One UnembedFn per layer, applying that layer's fitted affine map
    before the model's own frozen final norm + lm_head -- pass the returned
    list anywhere a per-layer unembed list is accepted
    (logit_lens.run_layer_script_masses/detect_detour), in place of
    make_unembed_fn's single raw-lens callable."""
    raw_unembed = make_unembed_fn(model)

    def build(w: torch.Tensor, b: torch.Tensor) -> UnembedFn:
        def fn(hidden: torch.Tensor) -> torch.Tensor:
            mapped = hidden.to(torch.float32) @ w + b
            return raw_unembed(mapped)

        return fn

    return [build(w, b) for w, b in lens_params]


def evaluate_tuned_lens(
    model, tokenizer, lens_params: LensParams, holdout_texts: list[str], device: str, max_length: int = 128
) -> list[dict]:
    """RQ2-follow-up EXP-1: nothing previously checked whether a fitted tuned
    lens is actually calibrated before its crossover-layer readout was
    trusted (RQ2-1's extreme layer-0 Qwen result was reported with no such
    check). Two per-layer diagnostics, both on text disjoint from the fit
    set: (a) R^2 of the affine map's prediction of the true final-layer
    hidden state -- near-perfect on the fit set but poor here would mean the
    map is overfit, not calibrated; (b) top-1 agreement between the lens's
    predicted next token at that layer and the model's own true top-1 token
    (obtained by reading the real final-layer hidden state through the same
    raw unembedding) -- a working tuned lens should climb from near-chance at
    layer 0 toward ~100% by the final layer, since target and prediction
    become identical there by construction. Flat or non-monotonic agreement
    is evidence the lens isn't doing what a tuned lens is supposed to do."""
    from urdu_interp import extraction

    raw_unembed = make_unembed_fn(model)
    per_text_layers = [
        extraction.extract_hidden_states(model, tokenizer, t, device=device, max_length=max_length).layers
        for t in holdout_texts
    ]
    num_layers = per_text_layers[0].shape[0]
    hidden_dim = per_text_layers[0].shape[-1]
    stacked = torch.cat([layers.reshape(num_layers, -1, hidden_dim) for layers in per_text_layers], dim=1)
    target = stacked[-1]
    true_top1 = raw_unembed(target).argmax(dim=-1)

    results = []
    for layer in range(num_layers):
        w, b = lens_params[layer]
        pred = stacked[layer].to(torch.float32) @ w + b
        ss_res = ((target - pred) ** 2).sum().item()
        ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        lens_top1 = raw_unembed(pred).argmax(dim=-1)
        top1_agreement = (lens_top1 == true_top1).float().mean().item()
        results.append({"layer": layer, "r2_holdout": r2, "top1_agreement": top1_agreement, "n_tokens": target.shape[0]})
    return results
