import numpy as np
from sklearn.metrics import roc_auc_score


def failure_detection_auroc(y_true, y_pred, uncertainty_scores) -> float:
    """Does higher uncertainty predict a higher chance the prediction is wrong?
    uncertainty_scores: higher = more uncertain (e.g. 1 - calibrated confidence, or conformal set size).
    """
    is_error = (np.asarray(y_true) != np.asarray(y_pred)).astype(int)
    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return float("nan")
    return roc_auc_score(is_error, uncertainty_scores)


def risk_coverage_curve(y_true, y_pred, uncertainty_scores):
    """Sort by ascending uncertainty; at each coverage level, report the error rate
    among the retained (most confident) predictions."""
    order = np.argsort(uncertainty_scores)
    y_true, y_pred = np.asarray(y_true)[order], np.asarray(y_pred)[order]
    n = len(y_true)
    coverages, risks = [], []
    for k in range(1, n + 1):
        coverages.append(k / n)
        risks.append(np.mean(y_true[:k] != y_pred[:k]))
    return np.array(coverages), np.array(risks)
