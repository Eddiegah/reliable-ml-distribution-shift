import numpy as np

from src.models.neural import (
    ensemble_predict,
    mc_dropout_predict,
    predict_proba,
    train_deep_ensemble,
    train_single_net,
)


def _toy_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


def test_train_single_net_returns_valid_probabilities():
    X, y = _toy_data()
    model = train_single_net(X, y, epochs=3, batch_size=64)
    probs = predict_proba(model, X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_deep_ensemble_members_have_different_weights():
    X, y = _toy_data()
    models = train_deep_ensemble(X, y, n_members=3, epochs=2, batch_size=64)
    mean_proba, per_member = ensemble_predict(models, X)
    assert per_member.shape == (3, len(X))
    assert mean_proba.shape == (len(X),)
    # different seeds/init should not converge to bit-identical outputs
    assert not np.allclose(per_member[0], per_member[1])


def test_mc_dropout_returns_multiple_stochastic_samples():
    X, y = _toy_data()
    model = train_single_net(X, y, epochs=2, batch_size=64)
    mean_proba, samples = mc_dropout_predict(model, X, n_samples=10)
    assert samples.shape == (10, len(X))
    assert mean_proba.shape == (len(X),)
    assert np.all((samples >= 0) & (samples <= 1))
