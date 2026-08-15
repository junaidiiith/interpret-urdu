"""Orchestration that combines corpora + alignment + extraction with the
convergence/logit-lens math -- the part of each experiment that's genuinely
non-trivial (looping over a corpus, extracting contextualized hidden states,
skipping items that don't align, stacking into tensors). Scripts call these
functions rather than re-implementing the loop themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from urdu_interp import alignment, extraction, fragmentation, logit_lens
from urdu_interp.corpora import AlignedTriple
from urdu_interp.tasks import NextWordItem


@dataclass
class WordVectorBatch:
    vecs_a: torch.Tensor  # (N, num_layers, hidden_dim)
    vecs_b: torch.Tensor  # (N, num_layers, hidden_dim)
    ids: list[str]  # "{triple_id}#{word_index}", unique per word pair -- lets
    # callers dedupe/union the same language's vectors across two different
    # lang-pair runs (RQ1-3's shared centering population) without re-running
    # the model.
    fragmentation_a: list[int]  # lang_a subword-piece count for that word (RQ1-1)
    stats: alignment.AlignmentStats


def extract_triple_word_vectors(
    model, tokenizer, triples: list[AlignedTriple], lang_a: str, lang_b: str, device: str, max_triples: int | None = None
) -> WordVectorBatch:
    """One contextualized forward pass per language per triple (not per
    isolated word), then pool each aligned word's subwords out of that
    context. Returns one row per successfully aligned and pooled word, plus
    alignment stats for the triples that had to be dropped (missing field or
    word-count mismatch -- see alignment.positional_word_pairs)."""
    if max_triples is not None:
        triples = triples[:max_triples]

    vecs_a: list[torch.Tensor] = []
    vecs_b: list[torch.Tensor] = []
    ids: list[str] = []
    fragmentation_a: list[int] = []
    dropped_triples = 0
    for triple in tqdm(triples, desc=f"word vectors ({lang_a}/{lang_b})"):
        text_a = getattr(triple, lang_a)
        text_b = getattr(triple, lang_b)
        if not text_a or not text_b:
            dropped_triples += 1
            continue
        if alignment.positional_word_pairs(text_a, text_b) is None:
            dropped_triples += 1
            continue

        hs_a = extraction.extract_hidden_states(model, tokenizer, text_a, device=device)
        hs_b = extraction.extract_hidden_states(model, tokenizer, text_b, device=device)
        word_vecs_a = extraction.word_span_vectors(hs_a)
        word_vecs_b = extraction.word_span_vectors(hs_b)
        frags = fragmentation.per_word_fragmentation(tokenizer, text_a)
        for i, (va, vb) in enumerate(zip(word_vecs_a, word_vecs_b)):
            if va is None or vb is None:
                continue
            vecs_a.append(va)
            vecs_b.append(vb)
            ids.append(f"{triple.id}#{i}")
            fragmentation_a.append(frags[i] if i < len(frags) else -1)

    stats = alignment.AlignmentStats(
        total_triples=len(triples),
        aligned_triples=len(triples) - dropped_triples,
        dropped_triples=dropped_triples,
        total_word_pairs=len(vecs_a),
    )
    if not vecs_a:
        raise ValueError("no word pairs survived alignment/extraction for this corpus/model")
    return WordVectorBatch(torch.stack(vecs_a), torch.stack(vecs_b), ids, fragmentation_a, stats)


def extract_triple_sentence_vectors(
    model, tokenizer, triples: list[AlignedTriple], lang_a: str, lang_b: str, device: str, max_triples: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Whole-sentence/verse pooled vectors for corpora that are independently
    authored translations rather than transliterations (FLORES, eBible, OPUS
    pivot) -- no word-level alignment is attempted there, see alignment.py."""
    if max_triples is not None:
        triples = triples[:max_triples]

    vecs_a: list[torch.Tensor] = []
    vecs_b: list[torch.Tensor] = []
    kept_ids: list[str] = []
    for triple in tqdm(triples, desc=f"sentence vectors ({lang_a}/{lang_b})"):
        text_a = getattr(triple, lang_a)
        text_b = getattr(triple, lang_b)
        if not text_a or not text_b:
            continue
        hs_a = extraction.extract_hidden_states(model, tokenizer, text_a, device=device)
        hs_b = extraction.extract_hidden_states(model, tokenizer, text_b, device=device)
        vecs_a.append(extraction.sentence_vector(hs_a))
        vecs_b.append(extraction.sentence_vector(hs_b))
        kept_ids.append(triple.id)

    if not vecs_a:
        raise ValueError("no sentence pairs survived extraction for this corpus/model")
    return torch.stack(vecs_a), torch.stack(vecs_b), kept_ids


def fragmentation_covariate(tokenizer, triples: list[AlignedTriple], lang: str, max_triples: int | None = None) -> list[float]:
    if max_triples is not None:
        triples = triples[:max_triples]
    return [
        fragmentation.fragmentation_ratio(tokenizer, getattr(t, lang))
        for t in triples
        if getattr(t, lang)
    ]


def run_detour_for_items(
    model,
    tokenizer,
    items: list[NextWordItem],
    unembed_fn,
    target: str,
    competitors: list[str],
    device: str,
    top_k: int = 50,
    max_items: int | None = None,
    latin_subtype_fn=None,
) -> list[tuple[NextWordItem, logit_lens.DetourResult]]:
    if max_items is not None:
        items = items[:max_items]
    # tokenizer.convert_ids_to_tokens() returns the raw byte-level-BPE vocab
    # string (e.g. Urdu 'مح' round-trips through convert_ids_to_tokens as the
    # unrelated-looking 'ÙħØŃ' for byte-level BPE tokenizers like Qwen's) --
    # script classification needs the actually-decoded text instead.
    def decode_id(token_id: int) -> str:
        return tokenizer.decode([token_id])

    results = []
    for item in tqdm(items, desc="logit-lens detour"):
        hs = extraction.extract_hidden_states(model, tokenizer, item.prefix, device=device)
        detour = logit_lens.detect_detour(
            hs, unembed_fn, decode_id, target=target, competitors=competitors, top_k=top_k, latin_subtype_fn=latin_subtype_fn
        )
        results.append((item, detour))
    return results


def run_detour_over_generation(
    model,
    tokenizer,
    item: NextWordItem,
    unembed_fn,
    target: str,
    competitors: list[str],
    device: str,
    n_steps: int = 5,
    top_k: int = 50,
    latin_subtype_fn=None,
) -> list[logit_lens.DetourResult]:
    """RQ2-5: run the logit-lens detour at each of the first `n_steps`
    *generated* positions, not just the prompt's final position -- the
    single-position result is the thinnest possible slice of evidence for a
    claim about a "detour" through the network. Greedily decodes one token
    at a time (argmax logit) and re-extracts hidden states on the growing
    prefix at each step, reusing the same extraction/detection path as the
    single-position case rather than a separate generation implementation."""
    import torch

    prefix = item.prefix
    results = []
    for _ in range(n_steps):
        hs = extraction.extract_hidden_states(model, tokenizer, prefix, device=device)
        detour = logit_lens.detect_detour(
            hs, unembed_fn, lambda tid: tokenizer.decode([tid]), target=target, competitors=competitors,
            top_k=top_k, latin_subtype_fn=latin_subtype_fn,
        )
        results.append(detour)

        enc = tokenizer(prefix, return_tensors="pt").to(device)
        with torch.no_grad():
            next_logits = model(**enc).logits[0, -1, :]
        next_id = int(torch.argmax(next_logits).item())
        next_token = tokenizer.decode([next_id])
        prefix = prefix + next_token
    return results
