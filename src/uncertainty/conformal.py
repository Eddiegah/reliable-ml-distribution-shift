import numpy as np
from mapie.classification import SplitConformalClassifier


def split_conformal_fit(
    fitted_model, X_cal, y_cal, alpha: float = 0.1, method: str = "lac"
) -> SplitConformalClassifier:
    """MAPIE >= 1.0 API: the estimator must already be fitted (prefit=True);
    `conformalize` plays the role of the old cv="prefit" .fit() call."""
    conformal = SplitConformalClassifier(
        estimator=fitted_model,
        confidence_level=1 - alpha,
        conformity_score=method,
        prefit=True,
    )
    conformal.conformalize(X_cal, y_cal)
    return conformal


def split_conformal_predict(conformal: SplitConformalClassifier, X):
    y_pred, y_pred_sets = conformal.predict_set(X)
    return y_pred, y_pred_sets


def empirical_coverage(y_true, y_pred_sets) -> float:
    """y_pred_sets: boolean array of shape (n_samples, n_classes, n_confidence_levels)
    from split_conformal_predict, at the single confidence level used to fit."""
    y_true = np.asarray(y_true)
    covered = y_pred_sets[np.arange(len(y_true)), y_true, 0]
    return float(covered.mean())


def prediction_set_size(y_pred_sets) -> np.ndarray:
    return y_pred_sets[:, :, 0].sum(axis=1)
