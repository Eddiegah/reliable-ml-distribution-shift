import numpy as np

from src.evaluation.metrics import (
    expected_calibration_error,
    performance_degradation,
    performance_report,
)


def test_perfectly_calibrated_predictions_have_zero_ece():
    y_true = np.array([0, 1] * 50)
    y_prob = np.array([0.0, 1.0] * 50)
    assert expected_calibration_error(y_true, y_prob) == 0.0


def test_maximally_overconfident_predictions_have_high_ece():
    y_true = np.zeros(50)
    y_prob = np.ones(50)  # always predicts positive with full confidence, always wrong
    assert expected_calibration_error(y_true, y_prob) > 0.9


def test_performance_report_contains_expected_keys():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(200)
    report = performance_report(y_true, y_prob)
    for key in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier", "ece"):
        assert key in report


def test_performance_degradation_is_in_minus_shifted():
    in_dist = {"auroc": 0.9, "brier": 0.1}
    shifted = {"auroc": 0.7, "brier": 0.2}
    degradation = performance_degradation(in_dist, shifted)
    assert degradation["auroc"] == 0.9 - 0.7
    assert degradation["brier"] == 0.1 - 0.2
