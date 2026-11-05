from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def _resolve_pretrained_path(hf_repo: str) -> str:
    """Hub ids pass through; relative local checkpoint paths resolve against the repo root."""
    path = Path(hf_repo)
    if path.is_absolute():
        return str(path)
    local = paths.REPO_ROOT / path
    if local.exists():
        return str(local)
    return hf_repo


def load_model_and_tokenizer(spec: ModelSpec, device: str = "cuda", dtype=None):
    """Loads a causal LM + tokenizer for a registry entry, with hidden-state
    output enabled (required by extraction.py and logit_lens.py)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype if dtype is not None else torch.bfloat16
    pretrained = _resolve_pretrained_path(spec.hf_repo)
    revision = None if Path(pretrained).exists() and spec.revision is None else spec.revision
    # Local checkpoints (RQ3 LoRA merges) must not pass a Hub revision.
    kwargs: dict = {"pretrained_model_name_or_path": pretrained}
    if revision is not None:
        kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(**kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        **kwargs,
        torch_dtype=dtype,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
