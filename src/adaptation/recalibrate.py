from src.evaluation.metrics import performance_report
from src.uncertainty.calibration import calibrate, predict_proba_positive


def recalibrate_on_target(fitted_model, X_adapt, y_adapt, method: str = "isotonic"):
    """Phase 5: recalibrate using a small labeled sample from the shifted period.
    X_adapt/y_adapt must be held out from the final test set (see configs/default.yaml)."""
    return calibrate(fitted_model, X_adapt, y_adapt, method=method)


def compare_before_after(model, recalibrated_model, X_test, y_test) -> dict:
    before = performance_report(y_test, predict_proba_positive(model, X_test))
    after = performance_report(y_test, predict_proba_positive(recalibrated_model, X_test))
    return {"before": before, "after": after}
