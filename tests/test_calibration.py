import numpy as np
from sklearn.linear_model import LogisticRegression

from src.uncertainty.calibration import calibrate, predict_proba_positive


def _toy_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_calibrate_returns_valid_probabilities():
    X, y = _toy_data()
    split = len(X) // 2
    model = LogisticRegression().fit(X[:split], y[:split])

    calibrated = calibrate(model, X[split:], y[split:], method="isotonic")
    probs = predict_proba_positive(calibrated, X[split:])

    assert probs.shape[0] == len(X) - split
    assert np.all((probs >= 0) & (probs <= 1))
