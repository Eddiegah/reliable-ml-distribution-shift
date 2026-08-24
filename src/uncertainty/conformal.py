from mapie.classification import MapieClassifier


def split_conformal_fit(fitted_model, X_cal, y_cal, method: str = "score") -> MapieClassifier:
    mapie = MapieClassifier(estimator=fitted_model, cv="prefit", method=method)
    mapie.fit(X_cal, y_cal)
    return mapie


def split_conformal_predict(mapie: MapieClassifier, X, alpha: float = 0.1):
    y_pred, y_pred_sets = mapie.predict(X, alpha=alpha)
    return y_pred, y_pred_sets


def empirical_coverage(y_true, y_pred_sets) -> float:
    covered = [y_true[i] in y_pred_sets[i, :, 0].nonzero()[0] for i in range(len(y_true))]
    return sum(covered) / len(covered)
