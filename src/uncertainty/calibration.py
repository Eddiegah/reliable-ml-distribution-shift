from sklearn.calibration import CalibratedClassifierCV


def calibrate(fitted_model, X_cal, y_cal, method: str = "isotonic") -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def predict_proba_positive(model, X):
    return model.predict_proba(X)[:, 1]
