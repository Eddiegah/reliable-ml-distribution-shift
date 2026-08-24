from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, seed: int = 42) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_baselines(X_train, y_train, seed: int = 42) -> dict:
    return {
        "logistic_regression": train_logistic_regression(X_train, y_train, seed),
        "xgboost": train_xgboost(X_train, y_train, seed),
    }
