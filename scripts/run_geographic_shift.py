"""Geographic-shift extension (proposal Section 3): isolates geography as
the shift dimension, separate from the temporal shift in run_pipeline.py.
Single year (2023) throughout, so there's no year-over-year confound -
train on Northeast respondents, evaluate on a held-out Northeast slice
(in-region reference) versus the other three Census regions (out-of-region)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.geography import region_for_state_fips
from src.data.preprocess import NON_FEATURE_COLUMNS, impute_missing
from src.evaluation.metrics import performance_degradation, performance_report
from src.models.baselines import train_baselines
from src.uncertainty.calibration import calibrate, predict_proba_positive
from src.uncertainty.conformal import (
    empirical_coverage,
    split_conformal_fit,
    split_conformal_predict,
)
from src.uncertainty.failure_detection import failure_detection_auroc
from src.utils.config import load_config
from src.utils.seed import set_seed

YEAR = 2023
TRAIN_REGION = "Northeast"


def xy(df: pd.DataFrame, feature_cols: list[str]):
    return df[feature_cols].to_numpy(dtype=float), df["diabetes"].to_numpy(dtype=int)


def main():
    config = load_config()
    seed = config["model"]["seed"]
    set_seed(seed)

    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    df_year = df[df["survey_year"] == YEAR].copy()
    df_year["region"] = region_for_state_fips(df_year["state_fips"])
    df_year = df_year.dropna(subset=["region"])  # drops territories, not a US Census region

    train_region_df = df_year[df_year["region"] == TRAIN_REGION]
    train_df, holdout_df = train_test_split(train_region_df, test_size=0.25, random_state=seed)

    train_imp, medians = impute_missing(train_df, feature_cols=feature_cols)
    holdout_imp, _ = impute_missing(holdout_df, medians=medians, feature_cols=feature_cols)
    X_train, y_train = xy(train_imp, feature_cols)
    X_holdout, y_holdout = xy(holdout_imp, feature_cols)

    print(f"train({TRAIN_REGION})={len(y_train)}  holdout({TRAIN_REGION})={len(y_holdout)}")

    results = {"year": YEAR, "train_region": TRAIN_REGION, "by_model": {}}

    models = train_baselines(X_train, y_train, seed=seed)
    other_regions = [r for r in ("Northeast", "Midwest", "South", "West") if r != TRAIN_REGION]

    for name, model in models.items():
        holdout_report = performance_report(y_holdout, model.predict_proba(X_holdout)[:, 1])
        row = {"in_region_holdout": holdout_report, "out_of_region": {}}
        print(f"\n[{name}] in-region holdout ({TRAIN_REGION}) AUROC={holdout_report['auroc']:.4f}")
        for region in other_regions:
            region_df = df_year[df_year["region"] == region]
            region_imp, _ = impute_missing(region_df, medians=medians, feature_cols=feature_cols)
            X_r, y_r = xy(region_imp, feature_cols)
            report = performance_report(y_r, model.predict_proba(X_r)[:, 1])
            row["out_of_region"][region] = {
                "report": report,
                "degradation_vs_in_region": performance_degradation(holdout_report, report),
            }
            print(f"  -> {region}: AUROC={report['auroc']:.4f} ECE={report['ece']:.4f} (n={len(y_r)})")
        results["by_model"][name] = row

    # ---- calibration + conformal + failure detection for the primary model ----
    xgb = models["xgboost"]
    method = config["calibration"]["method"]
    calibrated = calibrate(xgb, X_holdout, y_holdout, method=method)
    alpha = config["conformal"]["alpha"]
    conformal = split_conformal_fit(xgb, X_holdout, y_holdout, alpha=alpha, method="lac")

    uncertainty_section = {}
    for region in [TRAIN_REGION] + other_regions:
        region_df = df_year[df_year["region"] == region]
        if region == TRAIN_REGION:
            region_df = holdout_df  # avoid re-using train rows for the reference region
        region_imp, _ = impute_missing(region_df, medians=medians, feature_cols=feature_cols)
        X_r, y_r = xy(region_imp, feature_cols)

        probs = predict_proba_positive(calibrated, X_r)
        y_pred = (probs >= 0.5).astype(int)
        uncertainty = 1 - 2 * np.abs(probs - 0.5)
        fd_auroc = failure_detection_auroc(y_r, y_pred, uncertainty)

        _, pred_sets = split_conformal_predict(conformal, X_r)
        coverage = empirical_coverage(y_r, pred_sets)

        uncertainty_section[region] = {
            "calibrated_report": performance_report(y_r, probs),
            "failure_detection_auroc": float(fd_auroc),
            "conformal_coverage": coverage,
        }
        print(f"[xgboost calibrated on {TRAIN_REGION} holdout] {region}: "
              f"ECE={uncertainty_section[region]['calibrated_report']['ece']:.4f}  "
              f"coverage={coverage:.4f}  failure-detection AUROC={fd_auroc:.4f}")

    results["uncertainty_calibrated_on_train_region_holdout"] = uncertainty_section

    Path("reports/geographic_shift_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to reports/geographic_shift_results.json")


if __name__ == "__main__":
    main()
