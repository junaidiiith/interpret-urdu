# interpret-urdu

Mechanistic-interpretability toolkit and experiment pipeline for the paper
*"Do Low-Resource Languages Have a Place of Their Own in Large Language
Models? An Empirical Study of Urdu and Hindi"* (Computational Humanities
Research, CHR 2027).

The library looks *inside* open-weight LLMs — not just at their output — to
ask whether Urdu gets a representational space of its own, or is silently
processed through English's (or Hindi's) machinery. It combines three
methods (representational convergence, logit-lens/causal patching, and
fine-tuning) into one reusable, model-agnostic pipeline, and is designed so
the same methodology can be pointed at other low-resource language pairs.

## What's in this repo

| Path | Contents |
|---|---|
| `src/urdu_interp/` | The library: corpus loaders, alignment, hidden-state extraction, convergence math, logit lens / tuned lens, causal patching, concept probes, LoRA fine-tuning, stats, reporting. |
| `scripts/` | One CLI entry point per experiment; each script wires the library functions together and writes its output to a local `results/` directory. |
| `tests/` | 71 offline, mocked unit tests — no model downloads or GPU required. |
| `configs/models.yaml` | The model registry (HF repo + pinned revision per model key) and the LoRA fine-tuned checkpoints. |
| `configs/words.yaml` | Curated Urdu ghazal vocabulary + a frequency-matched control word set, with literature citations. |

Corpora and experiment output aren't checked into git (fetched/generated
locally instead — see `scripts/prepare_data.py` and the RQ scripts below).

## Installation

Requires Python >= 3.11 and (for anything beyond `tests/`) a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Gated models (`gemma3n-e2b-instruct`, `gemma3n-e4b-base`) additionally need
a Hugging Face token with access accepted:

```bash
hf auth login
```

Verify the install with the offline test suite:

```bash
pytest
```

## Quickstart

```bash
# 1. Sanity-check the corpora (no GPU, ~1 min)
python scripts/prepare_data.py

# 2. RQ1 — does Urdu's representation converge with Hindi/English, layer by layer?
python scripts/run_rq1_convergence.py --model qwen3.5-0.8b-base \
    --corpus rekhta_ghazals --lang-pairs ur-hi ur-en --limit 2000 --device cuda:0

# 3. RQ2 — does the model transiently "detour" through Hindi/English tokens mid-generation?
python scripts/run_rq2_logit_lens.py --model qwen3.5-0.8b-base \
    --corpus ebible --lang ur --limit 1000 --device cuda:0

# 4. Aggregate results across every model that has RQ1/RQ2 output on disk
python scripts/analyze_all.py --lang-pair ur-hi --lang ur
```

Every script accepts `--model <key>` (a key from `configs/models.yaml`) and
`--device` (`cuda:0`, `cuda:1`, ... — each run is single-GPU, so independent
runs can be parallelized across GPUs). Model keys ending in `-urdu-ft` are
the LoRA fine-tuned checkpoints and are handled by the RQ3 scripts, not the
main RQ1/RQ2 pipeline.

## Using the library directly

The scripts are thin orchestration over `src/urdu_interp/`; the same
building blocks compose directly in Python:

```python
from urdu_interp import corpora, convergence, models, pipeline

spec = models.get_spec("qwen3.5-0.8b-base")
model, tokenizer = models.load_model_and_tokenizer(spec, device="cuda:0")

triples = corpora.load_rekhta(kind="ghazals", limit=200)
batch = pipeline.extract_triple_word_vectors(model, tokenizer, triples, lang_a="ur", lang_b="hi", device="cuda:0")

result = convergence.layerwise_convergence(batch.vecs_a, batch.vecs_b, n_null=30)
print(result.effect_size)  # per-layer effect size vs. a shuffled-pair null
```

Module boundaries worth knowing before extending the library:

- **`corpora.py` / `alignment.py`** — every loader relies on alignment that's
  exact by construction (shared poem/verse/line id), never a statistical
  aligner; `alignment.py` does word-level positional matching on top.
- **`extraction.py`** — the only place hidden states are pulled from a real
  model; everything downstream operates on plain tensors and is unit-testable
  with fixtures.
- **`convergence.py` (RQ1)**, **`logit_lens.py` / `tuned_lens.py` (RQ2)** —
  read a representation without touching model behavior.
- **`causal.py` / `concept_attribution.py`** — the causal counterpart: forward
  hooks that patch or steer the residual stream, so a "detour" found by the
  logit lens can be verified to actually affect generation, not just be
  present but inert.
- **`finetune.py`** — LoRA continued-pretraining (RQ3) plus a matching
  perplexity eval.
- **`pipeline.py`** — the non-trivial orchestration glue (loop a corpus,
  extract, skip misaligned items, stack into tensors) that scripts call
  instead of reimplementing.

## Experiments

The pipeline is organized around three research questions, each with a
dedicated script (or family of scripts) and CSV/JSON output under
`results/`:

**RQ1 — representational convergence** (`run_rq1_convergence.py`,
`results/rq1_convergence/`): does a model's hidden-state representation of
matched Urdu content converge, layer by layer, with the same content in
Hindi vs. English, and how early/strongly? Run per model x corpus
(`rekhta_ghazals`, `flores_dev`, `ebible`) x language pair (`ur-hi`,
`ur-en`), against a shuffled-pair null to get an effect size, not just a raw
similarity number. `run_rq1_negative_control.py` adds a negative control on
the measurement apparatus itself.

**RQ2 — the logit-lens "detour"** (`run_rq2_logit_lens.py`,
`results/rq2_logit_lens/`): applies the model's own final readout to
intermediate-layer hidden states to see whether the model transiently
"thinks in" Hindi- or English-scripted tokens before settling on Urdu
output, and at what layer that resolves. A cluster of follow-up experiments
tests whether this is real and causal, not a lens artifact:
- `validate_tuned_lens.py` (EXP-1) — is a fitted tuned lens actually better
  calibrated than the raw logit lens, or just different?
- `run_exp2_patchscopes.py` (EXP-2) — Patchscopes: patch the hidden state
  into a separate prompt and let the model's own trained computation
  generate a verbalization, sidestepping the lens's linearity assumption.
- `run_exp3_concept_probes.py` (EXP-3) — a word-repetition task with a single
  known-correct answer per item, replacing the lens's fuzzy per-token script
  vote.
- `run_exp4_causal_patching.py` / `run_exp5_steering_dose_response.py`
  (EXP-4/5) — causal intervention via forward hooks: swap in a borrowed
  hidden state, or add a steering vector, and check whether that actually
  changes the model's output (not just what's decodable but functionally
  inert).
- `run_exp6_behavioral_drift.py` (EXP-6) — does the detour show up in
  ordinary behavior, e.g. reasoning scratch-work defaulting to English?
- `run_exp7_persian_control.py` (EXP-7) — a same-script control (Persian)
  to separate "script effect" from "language effect."
- `run_exp8_concept_significance.py` / `concept_attribution.py` (EXP-8) —
  TCAV-style concept activation vectors with a permutation-null significance
  test, so a causal effect can be attributed specifically to an
  "English-direction" push rather than any similar-magnitude perturbation.
- `run_rq2_multiposition_detour.py` (RQ2-5) — checks the crossover-layer
  pattern is stable across nearby token positions, not a one-off.

**RQ3 — does fine-tuning on Urdu literature change the internal geometry?**
(`run_rq3_finetune.py` then `run_rq3_eval.py`, `results/rq3_finetune/`,
`results/finetuned_models/`): LoRA-continued-pretrain a base model on Rekhta
Urdu poetry, then rerun RQ1's convergence measurement pre/post fine-tuning
as a paired before/after comparison (not just a correlational one).
`run_rq3_perplexity_ci.py` adds a bootstrap confidence interval on held-out
perplexity.

**Cross-model aggregation and behavioral cross-checks**:
- `analyze_all.py` — pools every model's RQ1/RQ2 CSVs into cross-model
  summary tables and the RQ1-vs-RQ2 recovery-layer correlation the
  three-question design is built around.
- `run_behavioral_qa.py` (MOT-6) — a factual-QA behavioral cross-check
  (`data/qa/factual_qa.json`), asked in both Urdu and Hindi, since every
  other measurement here is representational/geometric rather than
  task-based.
- `analyze_register_drift.py` (MOT-2), `analyze_rekhta_alignment_bias.py`
  (DAT-1) — targeted checks on two of the study's own confounds: whether
  "same language, different script" overstates Hindi-Urdu closeness in
  formal registers, and whether the Rekhta word-alignment sample (a ~6.6%
  survivor pool) is a biased subsample.

Eight base/instruct models are registered across two architecture families
(`configs/models.yaml`): Gemma-3n (E2B-instruct, E4B-base), Qwen3.5
(0.8B/2B, base/instruct), and SmolLM3-3B (base/instruct), plus four LoRA
Urdu-fine-tuned variants for RQ3.

## Testing

```bash
pytest              # 71 tests, fully offline/mocked, no GPU or model download needed
```

## License

GPLv3 — see [LICENSE](LICENSE).
