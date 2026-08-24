import numpy as np

from src.models.baselines import train_baselines


def test_train_baselines_returns_fitted_models_with_predict_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    models = train_baselines(X, y, seed=0)

    assert set(models.keys()) == {"logistic_regression", "xgboost"}
    for model in models.values():
        probs = model.predict_proba(X)[:, 1]
        assert probs.shape[0] == len(X)
        assert np.all((probs >= 0) & (probs <= 1))
