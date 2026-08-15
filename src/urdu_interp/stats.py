from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def first_stable_index(flags: list[bool]) -> int | None:
    """Earliest index i such that flags[i:] are all True -- the 'once it
    turns on, it never turns back off' definition used for both RQ1's
    recovery layer and RQ2's native crossover layer, so the two are computed
    the same way and stay comparable."""
    n = len(flags)
    for i in range(n):
        if all(flags[i:]):
            return i
    return None


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns q-values in the same order
    as the input p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p-value down
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty(n)
    q[order] = q_ranked
    return q.tolist()


def paired_delta_test(before: list[float], after: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test on (after - before); returns (statistic, p-value)."""
    if len(before) != len(after):
        raise ValueError("before/after must be the same length")
    return scipy_stats.wilcoxon(after, before)


def block_bootstrap_paired_test(
    before: list[float], after: list[float], block_size: int = 4, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap test for the mean paired delta (after - before) that
    resamples *contiguous blocks* of layers instead of individual layers --
    adjacent network layers' effect sizes are strongly autocorrelated (a
    smooth curve, not independent draws), so a plain per-layer paired test
    (e.g. Wilcoxon) is pseudoreplication (STA-1). Returns
    (observed_mean_delta, two-sided p-value: fraction of block-bootstrap
    resamples whose mean delta is as far from zero as observed, in either
    direction, under a sign-flip null)."""
    if len(before) != len(after):
        raise ValueError("before/after must be the same length")
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    n = len(delta)
    if n == 0:
        return 0.0, 1.0
    observed = float(delta.mean())

    n_blocks = max(1, n // block_size)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        # sign-flip each contiguous block as a unit (not each layer
        # independently) -- the null is "no consistent before/after shift",
        # preserving each block's internal autocorrelation structure.
        starts = np.arange(0, n_blocks * block_size, block_size)
        signs = rng.choice([-1.0, 1.0], size=len(starts))
        flipped = delta.copy()
        for start, sign in zip(starts, signs):
            flipped[start : start + block_size] *= sign
        boot_means[i] = flipped.mean()

    p_value = float((np.abs(boot_means) >= abs(observed)).mean())
    return observed, p_value


def fit_piecewise_changepoint(y: list[float]) -> tuple[int, float]:
    """Best single breakpoint index k (1 <= k <= len(y)-1) splitting y into
    two segments [0:k) and [k:) such that fitting a separate OLS mean (flat
    segment) to each side minimizes total sum-of-squared-error -- a fitted
    alternative to assuming the network's exact midpoint is where an
    early/late-half split should fall (STA-4). Returns (best_k, min_sse).
    Brute-force over all candidate breakpoints; fine for the ~24-37 layers
    this project's models have."""
    arr = np.asarray(y, dtype=float)
    n = len(arr)
    if n < 2:
        return 0, 0.0

    def sse(segment: np.ndarray) -> float:
        if len(segment) == 0:
            return 0.0
        return float(((segment - segment.mean()) ** 2).sum())

    best_k, best_sse = 1, float("inf")
    for k in range(1, n):
        total = sse(arr[:k]) + sse(arr[k:])
        if total < best_sse:
            best_k, best_sse = k, total
    return best_k, best_sse


def permutation_test_diff_of_means(
    group_a: list[float], group_b: list[float], n_permutations: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Two-sided permutation test on the difference of means between two
    independent samples. Returns (observed_diff, p-value)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        diff = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return float(observed), float(p_value)
