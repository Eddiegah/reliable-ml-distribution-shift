import numpy as np

from src.uncertainty.failure_detection import failure_detection_auroc, risk_coverage_curve


def test_failure_detection_auroc_rewards_uncertainty_that_tracks_errors():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # wrong at indices 1, 3, 6, 7
    uncertainty = np.array([0.1, 0.9, 0.1, 0.8, 0.1, 0.1, 0.9, 0.8])  # high where wrong
    auroc = failure_detection_auroc(y_true, y_pred, uncertainty)
    assert auroc == 1.0


def test_risk_coverage_curve_is_monotonic_in_coverage():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    uncertainty = np.array([0.1, 0.2, 0.9, 0.3, 0.05])
    coverage, risk = risk_coverage_curve(y_true, y_pred, uncertainty)
    assert np.all(np.diff(coverage) > 0)
    assert len(coverage) == len(risk) == len(y_true)
