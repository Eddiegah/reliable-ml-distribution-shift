"""Phases 2-5 end to end: baselines -> shift evaluation -> calibration/conformal
uncertainty -> adaptation. Run after configs/default.yaml years are set and
data/processed/brfss_clean.parquet exists (built via src.data.preprocess.build_dataset).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptation.recalibrate import recalibrate_on_target
from src.data.preprocess import impute_missing, temporal_split
from src.evaluation.metrics import performance_degradation, performance_report
from src.models.baselines import train_baselines
from src.uncertainty.calibration import calibrate, predict_proba_positive
from src.uncertainty.conformal import (
    empirical_coverage,
    prediction_set_size,
    split_conformal_fit,
    split_conformal_predict,
)
from src.uncertainty.failure_detection import failure_detection_auroc
from src.utils.config import load_config
from src.utils.seed import set_seed


def xy(df: pd.DataFrame, feature_cols: list[str]):
    return df[feature_cols].to_numpy(dtype=float), df["diabetes"].to_numpy(dtype=int)


def main():
    config = load_config()
    set_seed(config["model"]["seed"])

    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in ("survey_year", "diabetes")]

    splits = temporal_split(
        df,
        train_years=config["data"]["train_years"],
        val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"],
        adaptation_sample_years=config["data"]["adaptation_sample_years"],
    )
    train_imputed, medians = impute_missing(splits["train"], feature_cols=feature_cols)
    val_imputed, _ = impute_missing(splits["val"], medians=medians, feature_cols=feature_cols)
    adapt_imputed, _ = impute_missing(splits["adapt"], medians=medians, feature_cols=feature_cols)
    test_imputed, _ = impute_missing(splits["test"], medians=medians, feature_cols=feature_cols)

    X_train, y_train = xy(train_imputed, feature_cols)
    X_val, y_val = xy(val_imputed, feature_cols)
    X_adapt, y_adapt = xy(adapt_imputed, feature_cols)
    X_test, y_test = xy(test_imputed, feature_cols)

    print(f"train(2017)={len(y_train)}  val(2019)={len(y_val)}  "
          f"adapt(2021)={len(y_adapt)}  test(2023)={len(y_test)}")

    results = {}

    # ---- Phase 2/3: baselines, in-distribution vs shifted ----
    models = train_baselines(X_train, y_train, seed=config["model"]["seed"])
    for name, model in models.items():
        val_report = performance_report(y_val, model.predict_proba(X_val)[:, 1])
        test_report = performance_report(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = {
            "val_2019_in_distribution": val_report,
            "test_2023_shifted_raw": test_report,
            "degradation_val_minus_test": performance_degradation(val_report, test_report),
        }
        print(f"\n[{name}] val(2019) AUROC={val_report['auroc']:.4f} ECE={val_report['ece']:.4f}"
              f"  |  test(2023) AUROC={test_report['auroc']:.4f} ECE={test_report['ece']:.4f}")

    # ---- Phase 4: calibration + conformal on the primary model (XGBoost) ----
    xgb = models["xgboost"]
    method = config["calibration"]["method"]
    calibrated = calibrate(xgb, X_val, y_val, method=method)

    cal_val_report = performance_report(y_val, predict_proba_positive(calibrated, X_val))
    cal_test_report = performance_report(y_test, predict_proba_positive(calibrated, X_test))
    results["xgboost"]["calibrated_on_val_2019"] = {
        "val_2019": cal_val_report,
        "test_2023_shifted": cal_test_report,
    }
    print(f"\n[xgboost + {method} calibration fit on val(2019)]"
          f"  val ECE={cal_val_report['ece']:.4f}  |  test(2023) ECE={cal_test_report['ece']:.4f}")

    alpha = config["conformal"]["alpha"]
    conformal = split_conformal_fit(xgb, X_val, y_val, alpha=alpha, method="lac")
    _, sets_val = split_conformal_predict(conformal, X_val)
    _, sets_test = split_conformal_predict(conformal, X_test)
    coverage_val = empirical_coverage(y_val, sets_val)
    coverage_test = empirical_coverage(y_test, sets_test)
    results["xgboost"]["conformal"] = {
        "target_coverage": 1 - alpha,
        "empirical_coverage_val_2019": coverage_val,
        "empirical_coverage_test_2023": coverage_test,
        "mean_set_size_test_2023": float(prediction_set_size(sets_test).mean()),
    }
    print(f"[xgboost + conformal, target={1-alpha:.0%}]"
          f"  val(2019) coverage={coverage_val:.4f}  |  test(2023) coverage={coverage_test:.4f}")

    # does high uncertainty on the shifted set actually predict errors?
    test_probs_calibrated = predict_proba_positive(calibrated, X_test)
    y_pred_test = (test_probs_calibrated >= 0.5).astype(int)
    uncertainty = 1 - 2 * np.abs(test_probs_calibrated - 0.5)  # 0 = confident, 1 = maximally unsure
    fd_auroc = failure_detection_auroc(y_test, y_pred_test, uncertainty)
    results["xgboost"]["failure_detection_auroc_test_2023"] = float(fd_auroc)
    print(f"[xgboost] failure-detection AUROC on test(2023) = {fd_auroc:.4f}"
          " (0.5 = uncertainty is uninformative, 1.0 = perfectly flags errors)")

    # ---- Phase 5: adaptation using the small 2021 sample ----
    recalibrated = recalibrate_on_target(xgb, X_adapt, y_adapt, method=method)
    recal_test_report = performance_report(y_test, predict_proba_positive(recalibrated, X_test))
    results["xgboost"]["recalibrated_on_adapt_2021"] = {"test_2023": recal_test_report}
    print(f"\n[xgboost + recalibrated on adapt(2021), evaluated on test(2023)]"
          f"  AUROC={recal_test_report['auroc']:.4f} ECE={recal_test_report['ece']:.4f}"
          f"  (vs raw ECE={results['xgboost']['test_2023_shifted_raw']['ece']:.4f},"
          f" vs val-calibrated ECE={cal_test_report['ece']:.4f})")

    with open("reports/phase2-5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to reports/phase2-5_results.json")


if __name__ == "__main__":
    main()
