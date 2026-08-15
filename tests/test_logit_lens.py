import torch

from urdu_interp.extraction import HiddenStates
from urdu_interp.logit_lens import (
    _resolve_final_norm,
    detect_detour,
    layer_script_mass,
    native_crossover_layer,
    run_layer_script_masses,
)


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_resolve_final_norm_plain_decoder_layout():
    norm = object()
    model = _Obj(model=_Obj(norm=norm))
    assert _resolve_final_norm(model) is norm


def test_resolve_final_norm_directly_on_model():
    norm = object()
    model = _Obj(norm=norm)
    assert _resolve_final_norm(model) is norm


def test_resolve_final_norm_one_level_language_model_layout():
    norm = object()
    model = _Obj(language_model=_Obj(norm=norm))
    assert _resolve_final_norm(model) is norm


def test_resolve_final_norm_gemma3n_nested_language_model_layout():
    # Gemma-3n: model.model is a Gemma3nModel wrapping the actual text
    # decoder as .language_model, which is what holds .norm.
    norm = object()
    model = _Obj(model=_Obj(language_model=_Obj(norm=norm)))
    assert _resolve_final_norm(model) is norm


def test_resolve_final_norm_returns_none_when_nothing_matches():
    model = _Obj(model=_Obj())
    assert _resolve_final_norm(model) is None

# a tiny fake vocabulary: id 0/1 are urdu-scripted tokens, 2/3 hindi, 4/5 latin
_VOCAB = {0: "م", 1: "حبت", 2: "मो", 3: "हब्बत", 4: "lo", 5: "ve"}


def _lookup(idx: int) -> str:
    return _VOCAB[idx]


def test_layer_script_mass_aggregates_topk_by_script():
    # put all mass on ids 0 and 1 (urdu)
    logits = torch.tensor([10.0, 10.0, -10.0, -10.0, -10.0, -10.0])
    mass = layer_script_mass(logits, _lookup, top_k=6)
    assert mass["urdu"] > 0.9
    assert mass["hindi"] < 0.05
    assert mass["latin"] < 0.05


def test_layer_script_mass_respects_top_k():
    # id 4/5 (latin) have the highest logits; with top_k=2 only they count
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0, 100.0])
    mass = layer_script_mass(logits, _lookup, top_k=2)
    assert mass["latin"] > 0.9
    assert mass["urdu"] == 0.0
    assert mass["hindi"] == 0.0


def test_native_crossover_layer_finds_earliest_permanent_lead():
    layer_masses = [
        {"urdu": 0.1, "hindi": 0.6, "latin": 0.1, "other": 0.2},
        {"urdu": 0.5, "hindi": 0.3, "latin": 0.1, "other": 0.1},
        {"urdu": 0.7, "hindi": 0.1, "latin": 0.1, "other": 0.1},
    ]
    layer = native_crossover_layer(layer_masses, target="urdu", competitors=["hindi", "latin"])
    assert layer == 1


def test_native_crossover_layer_none_if_target_never_wins():
    layer_masses = [
        {"urdu": 0.1, "hindi": 0.8, "latin": 0.1},
        {"urdu": 0.2, "hindi": 0.7, "latin": 0.1},
    ]
    assert native_crossover_layer(layer_masses, target="urdu", competitors=["hindi", "latin"]) is None


def _make_hidden_states(num_layers: int = 3) -> HiddenStates:
    layers = torch.zeros(num_layers, 1, 4)
    return HiddenStates(layers=layers, offsets=[(0, 1)], text="x")


def test_run_layer_script_masses_calls_unembed_once_per_layer():
    scripted_logits = [
        torch.tensor([10.0, 10.0, -10.0, -10.0, -10.0, -10.0]),  # urdu-heavy
        torch.tensor([-10.0, -10.0, 10.0, 10.0, -10.0, -10.0]),  # hindi-heavy
        torch.tensor([10.0, 10.0, -10.0, -10.0, -10.0, -10.0]),  # urdu-heavy again
    ]
    it = iter(scripted_logits)

    def fake_unembed(hidden):
        return next(it)

    hs = _make_hidden_states(num_layers=3)
    masses = run_layer_script_masses(hs, fake_unembed, _lookup, top_k=6)

    assert len(masses) == 3
    assert masses[0]["urdu"] > 0.9
    assert masses[1]["hindi"] > 0.9
    assert masses[2]["urdu"] > 0.9


def test_detect_detour_end_to_end_with_fake_unembed():
    scripted_logits = [
        torch.tensor([-10.0, -10.0, 10.0, 10.0, -10.0, -10.0]),  # hindi-heavy (detour)
        torch.tensor([-10.0, -10.0, 10.0, 10.0, -10.0, -10.0]),  # still hindi-heavy
        torch.tensor([10.0, 10.0, -10.0, -10.0, -10.0, -10.0]),  # settles on urdu
    ]
    it = iter(scripted_logits)

    def fake_unembed(hidden):
        return next(it)

    hs = _make_hidden_states(num_layers=3)
    result = detect_detour(hs, fake_unembed, _lookup, target="urdu", competitors=["hindi", "latin"], top_k=6)

    assert result.native_crossover_layer == 2
    assert len(result.layer_masses) == 3


def test_run_layer_script_masses_accepts_a_per_layer_unembed_list():
    # RQ2-1: a tuned lens needs a different affine map per layer -- a list
    # of callables, indexed by layer, should work the same way a single
    # shared callable does for the raw lens.
    per_layer_logits = [
        torch.tensor([10.0, 10.0, -10.0, -10.0, -10.0, -10.0]),  # urdu
        torch.tensor([-10.0, -10.0, 10.0, 10.0, -10.0, -10.0]),  # hindi
        torch.tensor([-10.0, -10.0, -10.0, -10.0, 10.0, 10.0]),  # latin
    ]
    fns = [(lambda hidden, l=logits: l) for logits in per_layer_logits]
    hs = _make_hidden_states(num_layers=3)
    masses = run_layer_script_masses(hs, fns, _lookup, top_k=6)
    assert masses[0]["urdu"] > 0.9
    assert masses[1]["hindi"] > 0.9
    assert masses[2]["latin"] > 0.9


_ROMAN_URDU_WORDS = frozenset({"kya"})


def test_layer_script_mass_breaks_down_latin_bucket_with_subtype_fn():
    vocab = {0: "the", 1: "kya"}  # id 0 -> english word, id 1 -> roman-urdu word

    def lookup(idx: int) -> str:
        return vocab[idx]

    def latin_subtype_fn(token: str) -> str:
        from urdu_interp.script_utils import classify_latin_subtype

        return classify_latin_subtype(token, _ROMAN_URDU_WORDS)

    logits = torch.tensor([10.0, 10.0])  # equal mass on both latin-script tokens
    mass = layer_script_mass(logits, lookup, top_k=2, latin_subtype_fn=latin_subtype_fn)
    assert mass["latin"] > 0.9  # both tokens are latin-script
    assert mass["latin_english"] > 0.4
    assert mass["latin_roman_urdu"] > 0.4
