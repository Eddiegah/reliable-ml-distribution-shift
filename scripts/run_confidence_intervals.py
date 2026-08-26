"""Bootstrap confidence intervals for the headline numbers reported across
Sections 4.1-4.4 of the paper. Retrains the (fast, CPU-only) classical
models fresh rather than reusing saved predictions, since none of the
earlier scripts persisted raw per-row predictions - retraining XGBoost
takes seconds, so this is simpler than plumbing prediction-caching through
every earlier script.

Does not cover Section 4.5 (deep ensemble): those predictions only exist
on the GPU session that produced them and weren't saved to disk, so a
bootstrap CI for that section needs a future GPU run that also persists
raw predictions - noted as a limitation rather than skipped silently.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.geography import region_for_state_fips
from src.data.preprocess import NON_FEATURE_COLUMNS, impute_missing, temporal_split
from src.evaluation.bootstrap import (
    bootstrap_independent_diff,
    bootstrap_metric,
    bootstrap_paired_diff,
    fast_auroc,
)
from src.evaluation.metrics import expected_calibration_error
from src.evaluation.subgroup import bin_age_group
from src.models.baselines import train_xgboost
from src.uncertainty.calibration import calibrate, predict_proba_positive
from src.uncertainty.conformal import split_conformal_fit
from src.uncertainty.weighted_conformal import (
    conformal_threshold,
    estimate_shift_weights,
    lac_scores,
    prediction_sets_from_threshold,
)
from src.utils.config import load_config
from src.utils.seed import set_seed

# max_n=None: fast_auroc and the other metrics below are all cheap enough
# (no sklearn per-call overhead) to bootstrap on the full evaluation set
# directly, so every point estimate and CI here uses all the real data.
MAX_N = None
N_BOOT = 300


def xy(df, feature_cols):
    return df[feature_cols].to_numpy(dtype=float), df["diabetes"].to_numpy(dtype=int)


def mapie_coverage_indicator(pred_sets: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """pred_sets: MAPIE's 3D (n_samples, n_classes, n_confidence_levels) format."""
    return pred_sets[np.arange(len(y_true)), y_true, 0].astype(float)


def weighted_coverage_indicator(pred_sets: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """pred_sets: weighted_conformal.py's 2D (n_samples, n_classes) format -
    a different shape from MAPIE's, so this is deliberately a separate
    function rather than reusing mapie_coverage_indicator on the wrong shape."""
    return pred_sets[np.arange(len(y_true)), y_true].astype(float)


def fd_auroc_metric(y_true, packed):
    y_pred, uncertainty = packed[:, 0].astype(int), packed[:, 1]
    is_error = (y_true != y_pred).astype(int)
    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return 0.5
    return fast_auroc(is_error, uncertainty)


def main():
    config = load_config()
    seed = config["model"]["seed"]
    set_seed(seed)
    results = {}

    # ---------------------------------------------------------------
    # Section 4.1: temporal shift
    # ---------------------------------------------------------------
    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    splits = temporal_split(
        df, train_years=config["data"]["train_years"], val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"], adaptation_sample_years=config["data"]["adaptation_sample_years"],
    )
    train_imp, medians = impute_missing(splits["train"], feature_cols=feature_cols)
    val_imp, _ = impute_missing(splits["val"], medians=medians, feature_cols=feature_cols)
    test_imp, _ = impute_missing(splits["test"], medians=medians, feature_cols=feature_cols)
    X_train, y_train = xy(train_imp, feature_cols)
    X_val, y_val = xy(val_imp, feature_cols)
    X_test, y_test = xy(test_imp, feature_cols)

    xgb = train_xgboost(X_train, y_train, seed=seed)
    val_probs = xgb.predict_proba(X_val)[:, 1]
    test_probs = xgb.predict_proba(X_test)[:, 1]

    ci_auroc_val = bootstrap_metric(y_val, val_probs, fast_auroc, n_boot=N_BOOT, max_n=MAX_N, seed=seed)
    ci_auroc_test = bootstrap_metric(y_test, test_probs, fast_auroc, n_boot=N_BOOT, max_n=MAX_N, seed=seed)
    ci_auroc_gap = bootstrap_independent_diff(
        y_val, val_probs, y_test, test_probs, fast_auroc, n_boot=N_BOOT, max_n=MAX_N, seed=seed
    )
    print(f"[4.1] XGBoost AUROC val={ci_auroc_val['point']:.4f} "
          f"[{ci_auroc_val['ci_low']:.4f}, {ci_auroc_val['ci_high']:.4f}]  "
          f"test={ci_auroc_test['point']:.4f} [{ci_auroc_test['ci_low']:.4f}, {ci_auroc_test['ci_high']:.4f}]  "
          f"gap={ci_auroc_gap['point']:.4f} [{ci_auroc_gap['ci_low']:.4f}, {ci_auroc_gap['ci_high']:.4f}]")

    calibrated = calibrate(xgb, X_val, y_val, method=config["calibration"]["method"])
    cal_test_probs = predict_proba_positive(calibrated, X_test)
    ci_ece_test = bootstrap_metric(y_test, cal_test_probs, expected_calibration_error, n_boot=N_BOOT, max_n=MAX_N, seed=seed)
    print(f"[4.1] Calibrated ECE test={ci_ece_test['point']:.4f} "
          f"[{ci_ece_test['ci_low']:.4f}, {ci_ece_test['ci_high']:.4f}]")

    conformal = split_conformal_fit(xgb, X_val, y_val, alpha=config["conformal"]["alpha"], method="lac")
    _, val_sets = conformal.predict_set(X_val)
    _, test_sets = conformal.predict_set(X_test)
    val_covered = mapie_coverage_indicator(val_sets, y_val)
    test_covered = mapie_coverage_indicator(test_sets, y_test)
    ci_coverage_gap = bootstrap_independent_diff(
        y_val, val_covered, y_test, test_covered, lambda yt, s: s.mean(), n_boot=N_BOOT, max_n=MAX_N, seed=seed
    )
    print(f"[4.1] Conformal coverage gap (val-test)={ci_coverage_gap['point']:.4f} "
          f"[{ci_coverage_gap['ci_low']:.4f}, {ci_coverage_gap['ci_high']:.4f}]")

    y_pred_test = (cal_test_probs >= 0.5).astype(int)
    uncertainty_test = 1 - 2 * np.abs(cal_test_probs - 0.5)
    packed_test = np.column_stack([y_pred_test, uncertainty_test])
    ci_fd_auroc = bootstrap_metric(y_test, packed_test, fd_auroc_metric, n_boot=N_BOOT, max_n=MAX_N, seed=seed)
    print(f"[4.1] Failure-detection AUROC={ci_fd_auroc['point']:.4f} "
          f"[{ci_fd_auroc['ci_low']:.4f}, {ci_fd_auroc['ci_high']:.4f}]")

    results["temporal_shift"] = {
        "auroc_val": ci_auroc_val, "auroc_test": ci_auroc_test, "auroc_gap_val_minus_test": ci_auroc_gap,
        "calibrated_ece_test": ci_ece_test, "conformal_coverage_gap_val_minus_test": ci_coverage_gap,
        "failure_detection_auroc_test": ci_fd_auroc,
    }

    # ---------------------------------------------------------------
    # Section 4.2: subgroup fairness (age bands, test 2023)
    # ---------------------------------------------------------------
    age_bands = bin_age_group(test_imp["age_group"].to_numpy())
    band_probs = {}
    for band in ("18-44", "45-64", "65+"):
        mask = age_bands == band
        ci = bootstrap_metric(y_test[mask], cal_test_probs[mask], fast_auroc, n_boot=N_BOOT, max_n=MAX_N, seed=seed)
        band_probs[band] = ci
        print(f"[4.2] AUROC {band}={ci['point']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] (n={ci['n']})")

    mask_young, mask_old = age_bands == "18-44", age_bands == "65+"
    ci_age_gap = bootstrap_independent_diff(
        y_test[mask_young], cal_test_probs[mask_young], y_test[mask_old], cal_test_probs[mask_old],
        fast_auroc, n_boot=N_BOOT, max_n=MAX_N, seed=seed,
    )
    print(f"[4.2] AUROC gap 18-44 minus 65+ = {ci_age_gap['point']:.4f} "
          f"[{ci_age_gap['ci_low']:.4f}, {ci_age_gap['ci_high']:.4f}]")
    results["subgroup_fairness"] = {"auroc_by_age_band": band_probs, "auroc_gap_18_44_minus_65plus": ci_age_gap}

    # ---------------------------------------------------------------
    # Section 4.3/4.4: geographic shift + weighted conformal
    # ---------------------------------------------------------------
    df_year = df[df["survey_year"] == 2023].copy()
    df_year["region"] = region_for_state_fips(df_year["state_fips"])
    df_year = df_year.dropna(subset=["region"])
    train_region_df = df_year[df_year["region"] == "Northeast"]
    geo_train_df, geo_holdout_df = train_test_split(train_region_df, test_size=0.25, random_state=seed)
    geo_train_imp, geo_medians = impute_missing(geo_train_df, feature_cols=feature_cols)
    geo_holdout_imp, _ = impute_missing(geo_holdout_df, medians=geo_medians, feature_cols=feature_cols)
    X_geo_train, y_geo_train = xy(geo_train_imp, feature_cols)
    X_geo_holdout, y_geo_holdout = xy(geo_holdout_imp, feature_cols)

    geo_xgb = train_xgboost(X_geo_train, y_geo_train, seed=seed)

    geo_scores_cal = lac_scores(geo_xgb, X_geo_holdout, y_geo_holdout)
    unweighted_threshold = conformal_threshold(geo_scores_cal, alpha=config["conformal"]["alpha"])

    region_results, weighted_diffs, region_predictions = {}, {}, {}
    for region in ("Northeast", "Midwest", "South", "West"):
        region_df = geo_holdout_df if region == "Northeast" else df_year[df_year["region"] == region]
        region_imp, _ = impute_missing(region_df, medians=geo_medians, feature_cols=feature_cols)
        X_r, y_r = xy(region_imp, feature_cols)
        probs_r = geo_xgb.predict_proba(X_r)[:, 1]
        region_predictions[region] = (y_r, probs_r)
        ci_auroc_r = bootstrap_metric(y_r, probs_r, fast_auroc, n_boot=N_BOOT, max_n=MAX_N, seed=seed)

        unweighted_sets = prediction_sets_from_threshold(geo_xgb, X_r, unweighted_threshold)
        unweighted_covered = weighted_coverage_indicator(unweighted_sets, y_r)
        ci_coverage_r = bootstrap_metric(y_r, unweighted_covered, lambda yt, s: s.mean(), n_boot=N_BOOT, max_n=MAX_N, seed=seed)
        print(f"[4.3] {region} AUROC={ci_auroc_r['point']:.4f} [{ci_auroc_r['ci_low']:.4f}, {ci_auroc_r['ci_high']:.4f}]  "
              f"coverage={ci_coverage_r['point']:.4f} [{ci_coverage_r['ci_low']:.4f}, {ci_coverage_r['ci_high']:.4f}]")
        region_results[region] = {"auroc": ci_auroc_r, "unweighted_coverage": ci_coverage_r}

        if region != "Northeast":
            weights = estimate_shift_weights(X_geo_holdout, X_r, seed=seed)
            weighted_threshold = conformal_threshold(geo_scores_cal, alpha=config["conformal"]["alpha"], weights=weights)
            weighted_sets = prediction_sets_from_threshold(geo_xgb, X_r, weighted_threshold)
            weighted_covered = weighted_coverage_indicator(weighted_sets, y_r)
            ci_improve = bootstrap_paired_diff(
                y_r, weighted_covered, unweighted_covered, lambda yt, s: s.mean(),
                n_boot=N_BOOT, max_n=MAX_N, seed=seed,
            )
            print(f"[4.4] {region} weighted-minus-unweighted coverage = {ci_improve['point']:.4f} "
                  f"[{ci_improve['ci_low']:.4f}, {ci_improve['ci_high']:.4f}]")
            weighted_diffs[region] = ci_improve

    y_ne, probs_ne = region_predictions["Northeast"]
    y_south, probs_south = region_predictions["South"]
    ci_geo_auroc_gap = bootstrap_independent_diff(
        y_ne, probs_ne, y_south, probs_south, fast_auroc, n_boot=N_BOOT, max_n=MAX_N, seed=seed
    )
    print(f"[4.3] AUROC gap Northeast minus South = {ci_geo_auroc_gap['point']:.4f} "
          f"[{ci_geo_auroc_gap['ci_low']:.4f}, {ci_geo_auroc_gap['ci_high']:.4f}]")

    results["geographic_shift"] = {"by_region": region_results, "auroc_gap_northeast_minus_south": ci_geo_auroc_gap}
    results["weighted_conformal"] = {"coverage_improvement_by_region": weighted_diffs}

    with open("reports/confidence_intervals.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to reports/confidence_intervals.json")


if __name__ == "__main__":
    main()
