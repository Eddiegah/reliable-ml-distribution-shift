import numpy as np
from sklearn.linear_model import LogisticRegression

from src.uncertainty.weighted_conformal import (
    conformal_threshold,
    empirical_coverage,
    estimate_shift_weights,
    lac_scores,
    prediction_sets_from_threshold,
    weighted_quantile,
)


def test_weighted_quantile_matches_unweighted_when_weights_are_uniform():
    values = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    uniform_weights = np.ones(5)
    q_weighted = weighted_quantile(values, uniform_weights, 0.6)
    q_unweighted = float(np.quantile(values, 0.6, method="lower"))
    assert q_weighted == q_unweighted


def test_weighted_quantile_shifts_toward_heavily_weighted_values():
    values = np.array([0.1, 0.2, 0.3, 0.9])
    weights = np.array([1.0, 1.0, 1.0, 100.0])  # last point dominates
    q = weighted_quantile(values, weights, 0.5)
    assert q == 0.9


def test_estimate_shift_weights_are_higher_for_calibration_points_resembling_test():
    rng = np.random.default_rng(0)
    X_cal = np.vstack([rng.normal(0, 1, size=(200, 1)), rng.normal(5, 1, size=(200, 1))])
    X_test = rng.normal(5, 1, size=(200, 1))  # test domain resembles the second cal cluster
    weights = estimate_shift_weights(X_cal, X_test)
    assert weights[:200].mean() < weights[200:].mean()


def test_conformal_coverage_reaches_target_in_distribution():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X[:1000], y[:1000])

    X_cal, y_cal = X[1000:1500], y[1000:1500]
    X_test, y_test = X[1500:], y[1500:]

    scores_cal = lac_scores(model, X_cal, y_cal)
    threshold = conformal_threshold(scores_cal, alpha=0.1)
    pred_sets = prediction_sets_from_threshold(model, X_test, threshold)
    coverage = empirical_coverage(y_test, pred_sets)
    assert 0.85 <= coverage <= 0.99
