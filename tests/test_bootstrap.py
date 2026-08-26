import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.bootstrap import (
    bootstrap_independent_diff,
    bootstrap_metric,
    bootstrap_paired_diff,
    excludes_zero,
    fast_auroc,
)


def _proportion(y_true, y_score):
    return float(np.mean(y_score))


def test_bootstrap_metric_ci_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_score = rng.random(2000)
    result = bootstrap_metric(y_true, y_score, roc_auc_score, n_boot=200)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_bootstrap_metric_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)
    y_true_small = rng.integers(0, 2, size=200)
    y_score_small = y_true_small * 0.6 + rng.normal(0, 0.3, size=200)
    y_true_large = rng.integers(0, 2, size=20000)
    y_score_large = y_true_large * 0.6 + rng.normal(0, 0.3, size=20000)

    small = bootstrap_metric(y_true_small, y_score_small, roc_auc_score, n_boot=300)
    large = bootstrap_metric(y_true_large, y_score_large, roc_auc_score, n_boot=300)
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])


def test_bootstrap_paired_diff_is_zero_when_conditions_are_identical():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=1000)
    score = rng.random(1000)
    result = bootstrap_paired_diff(y_true, score, score, roc_auc_score, n_boot=200)
    assert result["point"] == 0.0
    assert not excludes_zero(result)


def test_bootstrap_metric_point_estimate_ignores_max_n():
    """max_n must only speed up the CI's resampling loop - never change the
    reported point estimate, which is always computed on the full data."""
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=50000)
    y_score = rng.random(50000)
    full = bootstrap_metric(y_true, y_score, roc_auc_score, n_boot=20, max_n=None, seed=1)
    subsampled = bootstrap_metric(y_true, y_score, roc_auc_score, n_boot=20, max_n=5000, seed=1)
    assert full["point"] == subsampled["point"]
    assert full["n"] == subsampled["n"] == 50000


def test_fast_auroc_matches_sklearn():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, size=3000)
    y_score = rng.random(3000)
    assert np.isclose(fast_auroc(y_true, y_score), roc_auc_score(y_true, y_score))


def test_bootstrap_independent_diff_detects_a_real_gap():
    rng = np.random.default_rng(3)
    n = 5000
    # group A: score is informative; group B: score is pure noise
    y_a = rng.integers(0, 2, size=n)
    score_a = y_a * 0.8 + rng.normal(0, 0.2, size=n)
    y_b = rng.integers(0, 2, size=n)
    score_b = rng.random(n)

    result = bootstrap_independent_diff(y_a, score_a, y_b, score_b, roc_auc_score, n_boot=300)
    assert result["point"] > 0.2
    assert excludes_zero(result)
