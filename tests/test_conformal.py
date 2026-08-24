import numpy as np
from sklearn.linear_model import LogisticRegression

from src.uncertainty.conformal import (
    empirical_coverage,
    prediction_set_size,
    split_conformal_fit,
    split_conformal_predict,
)


def _toy_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_split_conformal_reaches_roughly_target_coverage():
    X, y = _toy_data()
    train, cal, test = X[:200], X[200:300], X[300:]
    y_train, y_cal, y_test = y[:200], y[200:300], y[300:]

    model = LogisticRegression().fit(train, y_train)
    conformal = split_conformal_fit(model, cal, y_cal, alpha=0.1)
    _, y_sets = split_conformal_predict(conformal, test)

    coverage = empirical_coverage(y_test, y_sets)
    # split conformal gives a marginal, not exact, guarantee — allow slack
    assert coverage >= 0.75

    sizes = prediction_set_size(y_sets)
    assert sizes.shape[0] == len(y_test)
    assert np.all((sizes >= 0) & (sizes <= 2))
