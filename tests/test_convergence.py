import torch

from urdu_interp.convergence import (
    center,
    centered_cosine_per_layer,
    cosine_sim,
    effect_size_auc,
    layerwise_convergence,
    peak_layer_frac,
    recovery_layer,
    union_layer_mean,
)


def test_center_removes_column_mean():
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert torch.allclose(center(v).mean(dim=0), torch.zeros(2), atol=1e-6)


def test_cosine_sim_identical_vectors_is_one():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(cosine_sim(a, a), torch.ones(1))


def test_cosine_sim_orthogonal_vectors_is_zero():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(cosine_sim(a, b), torch.zeros(1), atol=1e-6)


def test_centered_cosine_per_layer_shape():
    vecs_a = torch.randn(5, 3, 4)
    vecs_b = torch.randn(5, 3, 4)
    out = centered_cosine_per_layer(vecs_a, vecs_b)
    assert out.shape == (5, 3)


def test_layerwise_convergence_perfect_match_gives_large_effect_size():
    torch.manual_seed(0)
    n, num_layers, h = 20, 1, 4
    vecs_a = torch.randn(n, num_layers, h)
    vecs_b = vecs_a.clone()

    result = layerwise_convergence(vecs_a, vecs_b, n_null=50, seed=1)

    assert result.n_pairs == n
    assert len(result.layer_scores) == 1
    assert result.per_item_scores.shape == (n, num_layers)
    assert torch.allclose(result.per_item_scores[:, 0], torch.ones(n), atol=1e-5)
    assert result.effect_size[0] > 1.5  # true pairs outscore the shuffled-pair null


def test_layerwise_convergence_output_length_matches_num_layers():
    vecs_a = torch.randn(8, 4, 5)
    vecs_b = torch.randn(8, 4, 5)
    result = layerwise_convergence(vecs_a, vecs_b, n_null=5, seed=0)
    assert len(result.layer_scores) == 4
    assert len(result.null_mean) == 4
    assert len(result.null_std) == 4
    assert len(result.effect_size) == 4


def test_recovery_layer_finds_earliest_stable_index():
    assert recovery_layer([0.1, 0.6, 0.7, 0.9], threshold=0.5) == 1


def test_recovery_layer_requires_staying_above_threshold_until_the_end():
    # index 1 dips back below threshold, so only index 2 onward is "stable"
    assert recovery_layer([0.6, 0.3, 0.9], threshold=0.5) == 2


def test_recovery_layer_none_if_it_never_stabilizes():
    assert recovery_layer([0.9, 0.1], threshold=0.5) is None


def test_peak_layer_frac_first_layer_is_zero():
    assert peak_layer_frac([0.9, 0.5, 0.1]) == 0.0


def test_peak_layer_frac_last_layer_is_one():
    assert peak_layer_frac([0.1, 0.5, 0.9]) == 1.0


def test_peak_layer_frac_middle_layer_is_fractional():
    assert peak_layer_frac([0.1, 0.9, 0.1, 0.1, 0.1]) == 0.25  # index 1 of 5 -> 1/4


def test_peak_layer_frac_single_layer_is_zero():
    assert peak_layer_frac([0.5]) == 0.0


def test_effect_size_auc_is_mean_of_effect_sizes():
    assert effect_size_auc([1.0, 2.0, 3.0]) == 2.0


def test_effect_size_auc_empty_is_zero():
    assert effect_size_auc([]) == 0.0


def test_union_layer_mean_dedupes_shared_ids_across_batches():
    # same id "a#0" appears in both batches with the same vector -- it
    # should only count once toward the mean.
    vecs_1 = torch.tensor([[[1.0, 1.0]], [[3.0, 3.0]]])  # ids: a#0, b#0
    ids_1 = ["a#0", "b#0"]
    vecs_2 = torch.tensor([[[1.0, 1.0]], [[5.0, 5.0]]])  # ids: a#0 (dup), c#0
    ids_2 = ["a#0", "c#0"]
    mean = union_layer_mean([(vecs_1, ids_1), (vecs_2, ids_2)])
    # union is {a#0: [1,1], b#0: [3,3], c#0: [5,5]} -> mean [3,3]
    assert torch.allclose(mean, torch.tensor([[3.0, 3.0]]))


def test_layerwise_convergence_with_external_mean_matches_manual_centering():
    torch.manual_seed(0)
    n, num_layers, h = 10, 2, 3
    vecs_a = torch.randn(n, num_layers, h)
    vecs_b = torch.randn(n, num_layers, h)
    mean_a = vecs_a.mean(dim=0)
    mean_b = vecs_b.mean(dim=0)

    with_external = layerwise_convergence(vecs_a, vecs_b, n_null=10, seed=0, mean_a=mean_a, mean_b=mean_b)
    with_default = layerwise_convergence(vecs_a, vecs_b, n_null=10, seed=0)
    # supplying the batch's own mean explicitly should reproduce the default
    # (self-computed-mean) behavior exactly.
    assert torch.allclose(with_external.per_item_scores, with_default.per_item_scores, atol=1e-5)


def test_layerwise_convergence_shared_mean_shifts_effect_size():
    torch.manual_seed(1)
    n, num_layers, h = 10, 1, 3
    vecs_a = torch.randn(n, num_layers, h)
    vecs_b = torch.randn(n, num_layers, h)
    # a deliberately different external mean should change the result vs.
    # the batch's own mean.
    skewed_mean = vecs_a.mean(dim=0) + 5.0
    default = layerwise_convergence(vecs_a, vecs_b, n_null=10, seed=0)
    skewed = layerwise_convergence(vecs_a, vecs_b, n_null=10, seed=0, mean_a=skewed_mean)
    assert not torch.allclose(default.per_item_scores, skewed.per_item_scores)
