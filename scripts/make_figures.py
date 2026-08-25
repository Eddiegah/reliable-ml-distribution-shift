"""Phase 6 support — the figures the proposal calls for (Section 6):
reliability diagrams and risk-coverage curves, plus ROC and a degradation
summary. Run after scripts/run_pipeline.py has produced fitted models is not
required — this script re-derives everything it needs from the same data
split so it can be run standalone."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from src.adaptation.recalibrate import recalibrate_on_target
from src.data.preprocess import impute_missing, temporal_split
from src.evaluation.metrics import performance_report
from src.models.baselines import train_baselines
from src.uncertainty.calibration import calibrate, predict_proba_positive
from src.uncertainty.failure_detection import risk_coverage_curve
from src.utils.config import load_config
from src.utils.seed import set_seed

FIG_DIR = Path("reports/figures")


def xy(df, feature_cols):
    return df[feature_cols].to_numpy(dtype=float), df["diabetes"].to_numpy(dtype=int)


def main():
    config = load_config()
    set_seed(config["model"]["seed"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in ("survey_year", "diabetes")]
    splits = temporal_split(
        df,
        train_years=config["data"]["train_years"],
        val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"],
        adaptation_sample_years=config["data"]["adaptation_sample_years"],
    )
    train_imp, medians = impute_missing(splits["train"], feature_cols=feature_cols)
    val_imp, _ = impute_missing(splits["val"], medians=medians, feature_cols=feature_cols)
    adapt_imp, _ = impute_missing(splits["adapt"], medians=medians, feature_cols=feature_cols)
    test_imp, _ = impute_missing(splits["test"], medians=medians, feature_cols=feature_cols)

    X_train, y_train = xy(train_imp, feature_cols)
    X_val, y_val = xy(val_imp, feature_cols)
    X_adapt, y_adapt = xy(adapt_imp, feature_cols)
    X_test, y_test = xy(test_imp, feature_cols)

    models = train_baselines(X_train, y_train, seed=config["model"]["seed"])

    # ---- Figure 1: ROC curves, val vs test, all three models ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in models.items():
        for X, y, split_label, style in [(X_val, y_val, "val 2019", "-"), (X_test, y_test, "test 2023", "--")]:
            probs = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probs)
            ax.plot(fpr, tpr, style, label=f"{name} ({split_label})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: in-distribution (2019) vs shifted (2023)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: reliability diagram, raw vs calibrated vs recalibrated (XGBoost, test 2023) ----
    xgb = models["xgboost"]
    method = config["calibration"]["method"]
    calibrated = calibrate(xgb, X_val, y_val, method=method)
    recalibrated = recalibrate_on_target(xgb, X_adapt, y_adapt, method=method)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="perfectly calibrated")
    for label, probs in [
        ("raw XGBoost", xgb.predict_proba(X_test)[:, 1]),
        ("calibrated on val 2019", predict_proba_positive(calibrated, X_test)),
        ("recalibrated on adapt 2021", predict_proba_positive(recalibrated, X_test)),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram on shifted test set (2023)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability_diagram.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: risk-coverage curve, XGBoost, test 2023 ----
    test_probs = predict_proba_positive(calibrated, X_test)
    y_pred = (test_probs >= 0.5).astype(int)
    uncertainty = 1 - 2 * np.abs(test_probs - 0.5)
    coverage, risk = risk_coverage_curve(y_test, y_pred, uncertainty)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(coverage, risk)
    ax.set_xlabel("Coverage (fraction of predictions retained, most-confident-first)")
    ax.set_ylabel("Error rate among retained predictions")
    ax.set_title("Risk-coverage curve: XGBoost on shifted test set (2023)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_coverage_curve.png", dpi=150)
    plt.close(fig)

    # ---- Figure 4: AUROC degradation summary, all three models ----
    fig, ax = plt.subplots(figsize=(6, 4))
    names, val_aurocs, test_aurocs = [], [], []
    for name, model in models.items():
        names.append(name)
        val_aurocs.append(performance_report(y_val, model.predict_proba(X_val)[:, 1])["auroc"])
        test_aurocs.append(performance_report(y_test, model.predict_proba(X_test)[:, 1])["auroc"])
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, val_aurocs, width, label="val 2019")
    ax.bar(x + width / 2, test_aurocs, width, label="test 2023")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("AUROC: in-distribution vs shifted, by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "auroc_degradation.png", dpi=150)
    plt.close(fig)

    print(f"Saved 4 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
