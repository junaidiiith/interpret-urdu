import pytest
import yaml

from urdu_interp.models import base_models, finetuned_models, get_spec, load_registry


def test_load_registry_parses_real_configs_models_yaml():
    registry = load_registry()
    assert len(registry) == 12  # 8 base/instruct checkpoints + 4 urdu-ft variants
    keys = {m.key for m in registry}
    assert "qwen3.5-0.8b-base" in keys
    assert "gemma3n-e2b-instruct" in keys


def test_base_models_excludes_finetuned_variants():
    registry = load_registry()
    base = base_models(registry)
    assert all(not m.is_finetuned for m in base)
    assert len(base) == 8


def test_finetuned_models_all_have_finetuned_from():
    registry = load_registry()
    ft = finetuned_models(registry)
    assert len(ft) == 4
    assert all(m.finetuned_from for m in ft)


def test_get_spec_looks_up_by_key():
    registry = load_registry()
    spec = get_spec("qwen3.5-0.8b-base", registry)
    assert spec.hf_repo == "Qwen/Qwen3.5-0.8B-Base"
    assert spec.gated is False


def test_get_spec_raises_keyerror_for_unknown_key():
    registry = load_registry()
    with pytest.raises(KeyError):
        get_spec("does-not-exist", registry)


def test_load_registry_from_explicit_fixture(tmp_path):
    fixture = {
        "models": [
            {
                "key": "fake-model",
                "role": "test_role",
                "hf_repo": "org/fake",
                "revision": "abc123",
                "gated": True,
                "variant": "instruct",
                "family": "fake-family",
            }
        ]
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(fixture), encoding="utf-8")

    registry = load_registry(path)
    assert len(registry) == 1
    assert registry[0].key == "fake-model"
    assert registry[0].gated is True
    assert registry[0].finetuned_from is None
