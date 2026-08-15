"""RQ2 follow-up EXP-2: Patchscopes (Ghandeharioun et al. 2024). Both the raw
and tuned logit lens force an intermediate hidden state through a *linear*
map (frozen unembedding, or a fitted affine map) the model was never trained
to be read out through at that layer -- RQ2-1's raw-vs-tuned disagreement is
a direct symptom of that shared assumption. This sidesteps it: patch the
hidden state into a *different*, fixed, simple prompt at the same layer and
let the model's own remaining forward computation -- its actual trained
computation, not an external probe -- generate a verbalization. There's no
map to calibrate, because the "readout" is the model's own decoding.

The target prompt reuses concept_probes.py's few-shot repeat-task framing
(establish "echo the word at this position back"), so the patched
position's completion is the model's own guess at "what word is this",
directly comparable in spirit to EXP-3's concept probes but read out by
generation instead of a lens.
"""
from __future__ import annotations

import torch

from urdu_interp import causal, extraction

_TARGET_TEMPLATE = "لفظ: پتھر → پتھر\nلفظ: کاغذ → کاغذ\nلفظ: چیز → "


def patch_and_verbalize(
    model, tokenizer, source_prefix: str, layer: int, device: str, max_new_tokens: int = 6, target_template: str = _TARGET_TEMPLATE
) -> str:
    """Extract `source_prefix`'s hidden state at `layer`, last position, and
    patch it into `target_template`'s last-token position for a single
    prefill forward pass -- then continue generation *unpatched* from that
    point (via explicit past_key_values continuation, not a hook left active
    for the whole model.generate() call, which would keep forcibly
    overwriting every newly generated token's own hidden state instead of
    seeding the representation once and letting the model take over)."""
    hs = extraction.extract_hidden_states(model, tokenizer, source_prefix, device=device)
    replacement = hs.layers[layer, -1, :]

    enc = tokenizer(target_template, return_tensors="pt").to(device)
    with causal.patch_hidden_state(model, layer=layer, position=-1, replacement=replacement):
        with torch.no_grad():
            prefill_out = model(**enc, use_cache=True)

    past_key_values = prefill_out.past_key_values
    next_id = int(torch.argmax(prefill_out.logits[0, -1, :]).item())
    generated_ids = [next_id]
    current_input = torch.tensor([[next_id]], device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_id = int(torch.argmax(out.logits[0, -1, :]).item())
            if next_id == tokenizer.eos_token_id:
                break
            generated_ids.append(next_id)
            current_input = torch.tensor([[next_id]], device=device)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
