from __future__ import annotations

from dataclasses import dataclass

import yaml

from urdu_interp import paths


@dataclass(frozen=True)
class ModelSpec:
    key: str
    role: str
    hf_repo: str
    revision: str | None
    gated: bool
    variant: str
    family: str
    finetuned_from: str | None = None

    @property
    def is_finetuned(self) -> bool:
        return self.finetuned_from is not None


def load_registry(path=paths.MODELS_YAML) -> list[ModelSpec]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [
        ModelSpec(
            key=m["key"],
            role=m["role"],
            hf_repo=m["hf_repo"],
            revision=m.get("revision"),
            gated=bool(m.get("gated", False)),
            variant=m["variant"],
            family=m["family"],
            finetuned_from=m.get("finetuned_from"),
        )
        for m in raw["models"]
    ]


def base_models(registry: list[ModelSpec] | None = None) -> list[ModelSpec]:
    registry = registry if registry is not None else load_registry()
    return [m for m in registry if not m.is_finetuned]


def finetuned_models(registry: list[ModelSpec] | None = None) -> list[ModelSpec]:
    registry = registry if registry is not None else load_registry()
    return [m for m in registry if m.is_finetuned]


def get_spec(key: str, registry: list[ModelSpec] | None = None) -> ModelSpec:
    registry = registry if registry is not None else load_registry()
    for m in registry:
        if m.key == key:
            return m
    raise KeyError(f"no model spec with key {key!r}")


def load_model_and_tokenizer(spec: ModelSpec, device: str = "cuda", dtype=None):
    """Loads a causal LM + tokenizer for a registry entry, with hidden-state
    output enabled (required by extraction.py and logit_lens.py)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype if dtype is not None else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_repo,
        revision=spec.revision,
        torch_dtype=dtype,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
