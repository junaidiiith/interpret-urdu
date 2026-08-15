"""RQ3: LoRA continued-pretraining on Urdu text, plus a matching perplexity
eval. chunk_texts()/rekhta_training_texts() are pure and unit-tested directly;
build_lora_model()/run_lora_finetune()/eval_perplexity() need a real model and
are exercised only by the scripts and the smoke test.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from urdu_interp.corpora import AlignedTriple


def rekhta_training_texts(triples: list[AlignedTriple], lang: str = "ur") -> list[str]:
    return [getattr(t, lang) for t in triples if getattr(t, lang)]


def chunk_texts(texts: list[str], tokenizer, block_size: int = 512) -> list[list[int]]:
    """Concatenate texts (EOS-separated) into one token stream, then split
    into fixed-size blocks for causal-LM training/eval."""
    eos_id = tokenizer.eos_token_id
    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
        if eos_id is not None:
            ids.append(eos_id)
    if len(ids) < block_size:
        return []
    n_blocks = len(ids) // block_size
    return [ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]


def build_lora_model(model, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05, target_modules=None):
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    return get_peft_model(model, config)


@dataclass
class FinetuneManifest:
    base_model_key: str
    output_dir: str
    num_blocks: int
    block_size: int
    lora_r: int
    lora_alpha: int
    epochs: int
    learning_rate: float
    seed: int = 0


def run_lora_finetune(
    model,
    tokenizer,
    texts: list[str],
    output_dir: str,
    base_model_key: str = "",
    block_size: int = 512,
    r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    seed: int = 0,
) -> FinetuneManifest:
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    blocks = chunk_texts(texts, tokenizer, block_size=block_size)
    if not blocks:
        raise ValueError("no training blocks produced from the given texts")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_dict({"input_ids": blocks})
    peft_model = build_lora_model(model, r=r, lora_alpha=lora_alpha)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        seed=seed,
    )
    trainer = Trainer(model=peft_model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    merged = peft_model.merge_and_unload()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest = FinetuneManifest(
        base_model_key=base_model_key,
        output_dir=output_dir,
        num_blocks=len(blocks),
        block_size=block_size,
        lora_r=r,
        lora_alpha=lora_alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    with open(Path(output_dir) / "finetune_manifest.json", "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
    return manifest


def eval_perplexity(model, tokenizer, texts: list[str], block_size: int = 512, device: str = "cuda") -> float:
    import torch

    blocks = chunk_texts(texts, tokenizer, block_size=block_size)
    if not blocks:
        raise ValueError("no evaluation blocks produced from the given texts")
    model.eval()
    losses = []
    with torch.no_grad():
        for block in blocks:
            input_ids = torch.tensor([block], device=device)
            out = model(input_ids=input_ids, labels=input_ids)
            losses.append(out.loss.item())
    return float(torch.exp(torch.tensor(sum(losses) / len(losses))))


@dataclass
class PerplexityResult:
    perplexity: float
    block_losses: list[float]
    n_blocks: int
    ci_low: float
    ci_high: float


def eval_perplexity_detailed(
    model, tokenizer, texts: list[str], block_size: int = 512, device: str = "cuda", n_boot: int = 2000, seed: int = 0
) -> PerplexityResult:
    """Same computation as eval_perplexity, but keeps the per-block losses
    (already computed, previously discarded) and adds a bootstrap confidence
    interval over blocks (ENG-3) -- a single scalar "perplexity dropped 32%"
    number with no attached uncertainty doesn't say whether that drop is
    real or within block-to-block noise."""
    import numpy as np
    import torch

    blocks = chunk_texts(texts, tokenizer, block_size=block_size)
    if not blocks:
        raise ValueError("no evaluation blocks produced from the given texts")
    model.eval()
    losses = []
    with torch.no_grad():
        for block in blocks:
            input_ids = torch.tensor([block], device=device)
            out = model(input_ids=input_ids, labels=input_ids)
            losses.append(out.loss.item())

    losses_arr = np.array(losses)
    perplexity = float(np.exp(losses_arr.mean()))

    rng = np.random.default_rng(seed)
    n = len(losses_arr)
    boot_ppls = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(losses_arr, size=n, replace=True)
        boot_ppls[i] = np.exp(sample.mean())
    ci_low, ci_high = np.percentile(boot_ppls, [2.5, 97.5])

    return PerplexityResult(perplexity=perplexity, block_losses=losses, n_blocks=n, ci_low=float(ci_low), ci_high=float(ci_high))
