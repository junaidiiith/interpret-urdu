import json

from urdu_interp.convergence import ConvergenceResult
from urdu_interp.logit_lens import DetourResult
from urdu_interp.reporting import convergence_rows, detour_rows, rows_to_dataframe, save_json, save_table


def _make_convergence_result():
    import torch

    return ConvergenceResult(
        n_pairs=10,
        layer_scores=[0.1, 0.5],
        null_mean=[0.05, 0.1],
        null_std=[0.02, 0.02],
        effect_size=[2.5, 20.0],
        per_item_scores=torch.zeros(10, 2),
    )


def test_convergence_rows_has_one_row_per_layer():
    rows = convergence_rows("model-x", "rekhta", "ur-hi", _make_convergence_result())
    assert len(rows) == 2
    assert rows[0]["model"] == "model-x"
    assert rows[0]["layer"] == 0
    assert rows[1]["effect_size"] == 20.0


def test_detour_rows_has_one_row_per_layer_with_mass_columns():
    result = DetourResult(
        layer_masses=[
            {"urdu": 0.1, "hindi": 0.6, "latin": 0.2, "other": 0.1},
            {"urdu": 0.7, "hindi": 0.1, "latin": 0.1, "other": 0.1},
        ],
        native_crossover_layer=1,
    )
    rows = detour_rows("model-x", "ebible", "ur", result)
    assert len(rows) == 2
    assert rows[1]["mass_urdu"] == 0.7
    assert rows[0]["native_crossover_layer"] == 1


def test_convergence_rows_includes_peak_layer_frac_and_effect_size_auc():
    rows = convergence_rows("model-x", "rekhta", "ur-hi", _make_convergence_result())
    assert rows[0]["peak_layer_frac"] == 1.0  # layer_scores=[0.1, 0.5] -> peak at last of 2 layers
    assert rows[0]["effect_size_auc"] == 11.25  # mean([2.5, 20.0])
    # constant across every row for the same (model, corpus, lang_pair)
    assert rows[0]["peak_layer_frac"] == rows[1]["peak_layer_frac"]


def test_detour_rows_flags_low_confidence_when_mass_other_is_high():
    result = DetourResult(
        layer_masses=[
            {"urdu": 0.3, "hindi": 0.3, "latin": 0.0, "other": 0.4},  # low confidence
            {"urdu": 0.7, "hindi": 0.1, "latin": 0.1, "other": 0.1},  # fine
        ],
        native_crossover_layer=1,
    )
    rows = detour_rows("model-x", "ebible", "ur", result)
    assert rows[0]["low_confidence"] is True
    assert rows[1]["low_confidence"] is False


def test_rows_to_dataframe_and_save_table_roundtrip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    df = rows_to_dataframe(rows)
    out = tmp_path / "sub" / "out.csv"
    save_table(df, out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").splitlines()[0] == "a,b"


def test_save_json_creates_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "out.json"
    save_json({"key": "قدر"}, out)
    assert json.loads(out.read_text(encoding="utf-8")) == {"key": "قدر"}
