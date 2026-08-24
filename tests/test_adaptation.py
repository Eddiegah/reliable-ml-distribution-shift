import numpy as np
from sklearn.linear_model import LogisticRegression

from src.adaptation.recalibrate import compare_before_after, recalibrate_on_target


def test_compare_before_after_reports_both_states():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    train, adapt, test = X[:150], X[150:220], X[220:]
    y_train, y_adapt, y_test = y[:150], y[150:220], y[220:]

    model = LogisticRegression().fit(train, y_train)
    recalibrated = recalibrate_on_target(model, adapt, y_adapt)

    comparison = compare_before_after(model, recalibrated, test, y_test)

    assert "before" in comparison and "after" in comparison
    assert "auroc" in comparison["before"] and "auroc" in comparison["after"]
