# interpret-urdu

Open-source mechanistic-interpretability toolkit for the paper:

> **Do Low-Resource Languages Have a Place of Their Own in Large Language Models? An Empirical Study of Urdu**  
> Computational Humanities Research (CHR) 2027

The library looks *inside* open-weight LLMs — not just at their outputs — to ask whether Urdu gets a representational space of its own, or is silently processed through English's (or Hindi's) machinery.

**Paper methods (no new experiments required to use this repo):**

| RQ | Question | Primary evidence in the paper |
|---|---|---|
| **RQ1** | Does Urdu's internal geometry default toward English, or stay closer to Hindi? | Layer-wise centered cosine convergence vs. a shuffled-pair null |
| **RQ2** | Is there a hidden English pathway, and what switches it on? | **Patchscopes** (observational) + **causal activation patching** with TCAV-style nulls; reasoning-scratch language check |
| **RQ3** | Does Urdu-poetry fine-tuning reshape the Urdu–Hindi relationship? | LoRA fine-tuning + before/after RQ1 geometry + held-out perplexity |

The methodology is language-agnostic: the same battery can be pointed at other low-resource, sibling-language script pairs with minimal changes.

**Paper model roster (reported results):** six open-weight checkpoints, two families, 0.8–3B parameters — Qwen3.5 (0.8B / 2B, base + instruct) and SmolLM3-3B (base + instruct). Gemma-3n entries remain in `configs/models.yaml` for optional extension, but are **not** part of the paper's reported family split.

---

## What's in this repo

| Path | Contents |
|---|---|
| `src/urdu_interp/` | Library: corpora, alignment, hidden-state extraction, convergence, Patchscopes, causal patching, concept attribution, LoRA fine-tuning, stats, reporting |
| `scripts/` | One CLI entry point per experiment; writes to local `results/` |
| `examples/` | Minimal quickstart scripts you can copy and adapt |
| `tests/` | Offline, mocked unit tests — no model download or GPU required |
| `configs/models.yaml` | Model registry (HF repo + pinned revision) and LoRA checkpoint keys |
| `configs/words.yaml` | Curated Urdu ghazal vocabulary + frequency-matched controls |

Corpora and experiment outputs are **not** in git (fetched/generated locally — see Quickstart).

---

## Installation

Requires Python ≥ 3.11. A CUDA GPU is needed for the model experiments; CPU is fine for `pytest` and corpus prep.

```bash
git clone https://github.com/junaidiiith/interpret-urdu.git
cd interpret-urdu
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest   # offline sanity check
```

Optional gated models (Gemma-3n only):

```bash
hf auth login
```

---

## Quickstart

### 1. Prepare corpora (no GPU, ~1 min)

```bash
python scripts/prepare_data.py --limit 200
```

This exercises Rekhta / FLORES / eBible loaders and writes `data/cache/corpus_summary.json`.

### 2. RQ1 — representational convergence (paper Step 3 / RQ1)

```bash
python scripts/run_rq1_convergence.py \
  --model qwen3.5-0.8b-base \
  --corpus rekhta_ghazals \
  --lang-pairs ur-hi ur-en \
  --limit 150 \
  --device cuda:0
```

Other paper corpora: `flores_dev`, `ebible`.  
Outputs land under `results/rq1_convergence/`.

### 3. RQ2 — Patchscopes + causal English pathway (paper headline for RQ2)

```bash
# Observational: does Urdu processing spontaneously verbalize as non-Urdu?
python scripts/run_exp2_patchscopes.py \
  --model qwen3.5-0.8b-base --limit 25 --device cuda:0

# Causal: does patching an English hidden state redirect real output logits?
python scripts/run_exp4_causal_patching.py \
  --model qwen3.5-0.8b-base --limit 40 --device cuda:0

# Concept-specific significance (TCAV-style null directions)
python scripts/run_exp8_concept_significance.py \
  --model qwen3.5-0.8b-base --device cuda:0

# When does English appear unforced? reasoning scratch-work vs final answer
python scripts/run_exp6_behavioral_drift.py \
  --model smollm3-3b-instruct --device cuda:0
```

> **Note:** Patchscopes / causal patching scripts are scoped to **Qwen3.5 and SmolLM3** (same as the paper). Gemma-3n's AltUp architecture is excluded from these hooks.

### 4. RQ3 — LoRA on Urdu poetry, then re-measure geometry

```bash
python scripts/run_rq3_finetune.py \
  --model qwen3.5-0.8b-base --device cuda:0

python scripts/run_rq3_eval.py \
  --model qwen3.5-0.8b-base-urdu-ft --device cuda:0
```

Fine-tuned keys use the `-urdu-ft` suffix; adapters are written under `results/finetuned_models/`.

### 5. One-file library example

```bash
python examples/quickstart_rq1.py --device cuda:0 --limit 50
```

Or in Python:

```python
from urdu_interp import corpora, convergence, models, pipeline

spec = models.get_spec("qwen3.5-0.8b-base")
model, tokenizer = models.load_model_and_tokenizer(spec, device="cuda:0")

triples = corpora.load_rekhta(kind="ghazals", limit=50)
batch = pipeline.extract_triple_word_vectors(
    model, tokenizer, triples, lang_a="ur", lang_b="hi", device="cuda:0"
)
result = convergence.layerwise_convergence(batch.vecs_a, batch.vecs_b, n_null=30)
print("effect_size_auc:", convergence.effect_size_auc(result.effect_size))
print("peak_layer_frac:", convergence.peak_layer_frac(result.layer_scores))
```

Every script accepts `--model <key>` from `configs/models.yaml` and `--device` (`cuda:0`, `cpu`, …). Independent runs can be parallelized across GPUs.

---

## Mapping scripts → paper

| Paper claim | Script(s) |
|---|---|
| RQ1 family split (Urdu↔Hindi / Urdu↔English geometry) | `run_rq1_convergence.py`, `run_rq1_negative_control.py` |
| RQ2 Patchscopes (no spontaneous English lean) | `run_exp2_patchscopes.py` |
| RQ2 causal English pathway | `run_exp4_causal_patching.py`, `run_exp5_steering_dose_response.py` |
| RQ2 TCAV / concept attribution | `run_exp8_concept_significance.py` |
| RQ2 reasoning-scratch English trigger | `run_exp6_behavioral_drift.py` |
| RQ3 LoRA + geometry / perplexity | `run_rq3_finetune.py`, `run_rq3_eval.py`, `run_rq3_perplexity_ci.py` |
| Cross-model aggregation | `analyze_all.py` |

Diagnostic / supporting scripts (logit lens, tuned lens, Persian control, Rekhta alignment bias, etc.) remain available under `scripts/` but are **not** the paper's headline evidence for RQ2 — the camera-ready paper prioritizes lens-free and causal methods.

---

## Extending to another language pair

1. Add a loader in `src/urdu_interp/corpora.py` that returns `AlignedTriple` rows (`id`, `ur`, `hi`, `en` — or rename fields and update call sites).
2. Prefer **transliteration-matched** text when isolating script (like Rekhta); use independent translations (like FLORES/eBible) when mixing script + style + lexis.
3. Reuse `pipeline.extract_triple_word_vectors` / `extract_triple_sentence_vectors`, then RQ1–RQ3 scripts unchanged aside from corpus name and language-pair flags.
4. For causal hooks, stick to architectures supported by `causal.py` (currently Qwen / SmolLM-style decoder stacks).

---

## Library map

| Module | Role |
|---|---|
| `corpora.py` / `alignment.py` | Data + exact positional word alignment (no statistical aligner) |
| `extraction.py` | Only place hidden states are read from a live model |
| `convergence.py` | RQ1 centered cosine + null effect sizes |
| `patchscopes.py` | RQ2 observational verbalization |
| `causal.py` / `concept_attribution.py` | RQ2 causal patching + TCAV-style nulls |
| `finetune.py` | RQ3 LoRA continued pretraining + perplexity |
| `pipeline.py` | Corpus → tensors orchestration used by scripts |
| `stats.py` | BH correction, block-bootstrap, permutation tests |

---

## Citation

If you use this toolkit, please cite the CHR paper (camera-ready / anthology entry once published) and this repository:

```bibtex
@inproceedings{ali2027urdu,
  title={Do Low-Resource Languages Have a Place of Their Own in Large Language Models? An Empirical Study of Urdu},
  author={Ali, Syed Juned},
  booktitle={Proceedings of Computational Humanities Research (CHR)},
  year={2027},
  url={https://github.com/junaidiiith/interpret-urdu}
}
```

---

## License

GPLv3 — see [LICENSE](LICENSE).
