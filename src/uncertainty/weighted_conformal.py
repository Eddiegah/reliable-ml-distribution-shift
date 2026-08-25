"""Weighted conformal prediction under covariate shift (Tibshirani, Barber,
Candes, & Ramdas, 2019). Standard split conformal assumes calibration and
test points are exchangeable; under covariate shift they aren't, so this
reweights calibration scores by an estimated density ratio w(x) = p_test(x) /
p_cal(x) before taking the quantile.

Simplification versus the paper's exact finite-sample construction: the
paper's weighted quantile technically depends on each test point's own
weight through the normalizing constant. With a calibration set in the tens
of thousands (our case), that single point's contribution to the
normalizer is negligible, so this implementation drops it and computes one
threshold shared across all test points - standard practice for calibration
sets at this scale, documented here rather than left implicit.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def estimate_shift_weights(X_cal: np.ndarray, X_test: np.ndarray, seed: int = 42) -> np.ndarray:
    """Trains a domain classifier (calibration=0, test=1) and returns the
    estimated density-ratio weight w(x) = P(test|x) / P(cal|x) for each
    calibration point."""
    X_domain = np.vstack([X_cal, X_test])
    y_domain = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X_domain, y_domain)
    p_test = np.clip(clf.predict_proba(X_cal)[:, 1], 1e-3, 1 - 1e-3)
    return p_test / (1 - p_test)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_weights = np.cumsum(weights)
    idx = np.searchsorted(cum_weights, quantile * cum_weights[-1])
    if idx >= len(values):
        return np.inf
    return float(values[idx])


def lac_scores(fitted_model, X, y) -> np.ndarray:
    """Least-ambiguous-set-style nonconformity score: 1 - predicted probability
    of the true class. Matches MAPIE's method="lac" used elsewhere in this repo."""
    p = fitted_model.predict_proba(X)
    return 1 - p[np.arange(len(y)), y]


def conformal_threshold(scores_cal: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones(len(scores_cal))
    return weighted_quantile(scores_cal, weights, 1 - alpha)


def prediction_sets_from_threshold(fitted_model, X, threshold: float) -> np.ndarray:
    p = fitted_model.predict_proba(X)
    scores = 1 - p  # shape (n_samples, n_classes)
    return scores <= threshold


def empirical_coverage(y_true: np.ndarray, pred_sets: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float(pred_sets[np.arange(len(y_true)), y_true].mean())


def mean_set_size(pred_sets: np.ndarray) -> float:
    return float(pred_sets.sum(axis=1).mean())
