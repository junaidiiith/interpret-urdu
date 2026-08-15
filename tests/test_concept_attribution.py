import torch

from urdu_interp.concept_attribution import fit_cav, fit_null_cavs, permutation_test


def test_fit_cav_recovers_a_clean_linear_separation():
    torch.manual_seed(0)
    # positive/negative clusters separated along dimension 0 only
    pos = torch.randn(30, 8) * 0.1 + torch.tensor([3.0, 0, 0, 0, 0, 0, 0, 0])
    neg = torch.randn(30, 8) * 0.1 + torch.tensor([-3.0, 0, 0, 0, 0, 0, 0, 0])
    cav = fit_cav(pos, neg, seed=0)
    assert cav.shape == (8,)
    assert torch.isclose(cav.norm(), torch.tensor(1.0), atol=1e-4)
    # the CAV should point overwhelmingly along dimension 0
    assert cav[0].abs() > 0.9


def test_fit_null_cavs_returns_requested_count_and_unit_norm():
    torch.manual_seed(1)
    pos = torch.randn(20, 6)
    neg = torch.randn(20, 6)
    nulls = fit_null_cavs(pos, neg, n_null=5, seed=0)
    assert len(nulls) == 5
    for cav in nulls:
        assert cav.shape == (6,)
        assert torch.isclose(cav.norm(), torch.tensor(1.0), atol=1e-4)


def test_fit_null_cavs_actually_shuffles_labels_not_a_noop():
    # Regression test for a real bug: an earlier label-permutation
    # implementation reconstructed the *original* label assignment
    # regardless of the permutation applied, silently making every "null"
    # CAV identical to the real CAV. With a strongly separable pos/neg
    # split, a null CAV fit on genuinely shuffled labels should classify
    # much worse on the original labels than the real CAV does.
    torch.manual_seed(3)
    pos = torch.randn(25, 10) * 0.1 + torch.tensor([5.0] + [0.0] * 9)
    neg = torch.randn(25, 10) * 0.1 + torch.tensor([-5.0] + [0.0] * 9)
    real_cav = fit_cav(pos, neg, seed=0)
    null_cavs = fit_null_cavs(pos, neg, n_null=8, seed=0)
    assert not any(torch.allclose(real_cav, null, atol=1e-3) for null in null_cavs)
    # the real CAV should align far more with the true separating axis
    # (dimension 0) than the average null CAV does
    real_alignment = real_cav[0].abs().item()
    null_alignment = sum(n[0].abs().item() for n in null_cavs) / len(null_cavs)
    assert real_alignment > null_alignment + 0.3


def test_fit_null_cavs_is_deterministic_given_a_seed():
    torch.manual_seed(2)
    pos = torch.randn(15, 5)
    neg = torch.randn(15, 5)
    a = fit_null_cavs(pos, neg, n_null=3, seed=42)
    b = fit_null_cavs(pos, neg, n_null=3, seed=42)
    for x, y in zip(a, b):
        assert torch.allclose(x, y)


def test_permutation_test_gives_small_p_value_when_real_effect_dominates():
    result = permutation_test(real_effect=0.9, null_effects=[0.01, 0.02, -0.01, 0.0, 0.03] * 6)  # n=30, none close to 0.9
    assert result["p_value"] < 0.05
    assert result["z_score"] > 3
    assert result["n_null"] == 30


def test_permutation_test_gives_large_p_value_when_real_effect_is_typical():
    null_effects = [0.1, 0.2, 0.15, 0.25, 0.3, 0.05, 0.4, 0.35, 0.12, 0.18]
    result = permutation_test(real_effect=0.2, null_effects=null_effects)
    assert result["p_value"] > 0.3


def test_permutation_test_p_value_never_zero():
    result = permutation_test(real_effect=100.0, null_effects=[0.0] * 10)
    assert result["p_value"] == 1 / 11
