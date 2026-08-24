from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate(fitted_model, X_cal, y_cal, method: str = "isotonic") -> CalibratedClassifierCV:
    """Calibrate an already-fitted model on a held-out slice (e.g. the validation
    or adaptation split) without retraining it. Requires scikit-learn >= 1.6
    (FrozenEstimator replaced the deprecated cv="prefit")."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def predict_proba_positive(model, X):
    return model.predict_proba(X)[:, 1]
