import numpy as np
from statsmodels.stats.multitest import multipletests

from urdu_interp.stats import (
    benjamini_hochberg,
    block_bootstrap_paired_test,
    fit_piecewise_changepoint,
    first_stable_index,
    paired_delta_test,
    permutation_test_diff_of_means,
)


def test_first_stable_index_basic():
    assert first_stable_index([False, True, True, True]) == 1
    assert first_stable_index([True, False, True]) == 2
    assert first_stable_index([False, False]) is None
    assert first_stable_index([True, True]) == 0


def test_benjamini_hochberg_matches_statsmodels():
    pvalues = [0.04, 0.02, 0.03, 0.5, 0.001, 0.2]
    ours = benjamini_hochberg(pvalues)
    _, expected, _, _ = multipletests(pvalues, method="fdr_bh")
    assert np.allclose(ours, expected, atol=1e-8)


def test_benjamini_hochberg_empty_input():
    assert benjamini_hochberg([]) == []


def test_paired_delta_test_detects_a_consistent_shift():
    before = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8, 1.0]
    after = [2.0, 2.1, 1.9, 2.05, 1.95, 2.2, 1.8, 2.0]
    _, p = paired_delta_test(before, after)
    assert p < 0.05


def test_paired_delta_test_length_mismatch_raises():
    try:
        paired_delta_test([1.0], [1.0, 2.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_permutation_test_separates_clearly_different_groups():
    group_a = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2]
    group_b = [0.0, 1.0, -1.0, 0.5, -0.5, 0.2]
    _, p = permutation_test_diff_of_means(group_a, group_b, n_permutations=500, seed=0)
    assert p < 0.01


def test_permutation_test_identical_groups_not_significant():
    group_a = [1.0, 2.0, 3.0, 4.0]
    group_b = [1.0, 2.0, 3.0, 4.0]
    _, p = permutation_test_diff_of_means(group_a, group_b, n_permutations=500, seed=0)
    assert p > 0.2


def test_block_bootstrap_paired_test_detects_a_consistent_shift():
    before = [1.0] * 24
    after = [2.0] * 24  # constant +1 shift, every block agrees
    delta, p = block_bootstrap_paired_test(before, after, block_size=4, n_boot=2000, seed=0)
    assert delta == 1.0
    assert p < 0.05


def test_block_bootstrap_paired_test_no_shift_not_significant():
    before = [1.0, 2.0, 3.0, 4.0] * 6
    after = [1.0, 2.0, 3.0, 4.0] * 6  # identical -- zero delta everywhere
    delta, p = block_bootstrap_paired_test(before, after, block_size=4, n_boot=2000, seed=0)
    assert delta == 0.0
    assert p == 1.0


def test_block_bootstrap_paired_test_length_mismatch_raises():
    try:
        block_bootstrap_paired_test([1.0], [1.0, 2.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_block_bootstrap_paired_test_empty_input():
    delta, p = block_bootstrap_paired_test([], [])
    assert delta == 0.0
    assert p == 1.0


def test_fit_piecewise_changepoint_finds_the_real_break():
    # flat at 0 for the first 5, flat at 10 for the last 5 -- the obvious
    # best split is at index 5.
    y = [0.0] * 5 + [10.0] * 5
    k, sse = fit_piecewise_changepoint(y)
    assert k == 5
    assert sse < 1e-6


def test_fit_piecewise_changepoint_constant_series_splits_anywhere_with_zero_sse():
    y = [3.0] * 8
    k, sse = fit_piecewise_changepoint(y)
    assert sse == 0.0
    assert 1 <= k <= 7


def test_fit_piecewise_changepoint_too_short_returns_zero():
    assert fit_piecewise_changepoint([1.0]) == (0, 0.0)
