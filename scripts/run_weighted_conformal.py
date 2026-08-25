"""Does weighted conformal prediction (Tibshirani et al., 2019) recover the
coverage that plain split conformal lost under geographic shift
(run_geographic_shift.py found South coverage dropped to 87.3%, vs a 90%
target)? Same train/holdout/region setup, same XGBoost model - only the
conformal step changes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.geography import region_for_state_fips
from src.data.preprocess import NON_FEATURE_COLUMNS, impute_missing
from src.models.baselines import train_xgboost
from src.uncertainty.weighted_conformal import (
    conformal_threshold,
    empirical_coverage,
    estimate_shift_weights,
    lac_scores,
    mean_set_size,
    prediction_sets_from_threshold,
)
from src.utils.config import load_config
from src.utils.seed import set_seed

YEAR = 2023
TRAIN_REGION = "Northeast"


def xy(df, feature_cols):
    return df[feature_cols].to_numpy(dtype=float), df["diabetes"].to_numpy(dtype=int)


def main():
    config = load_config()
    seed = config["model"]["seed"]
    set_seed(seed)
    alpha = config["conformal"]["alpha"]

    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    df_year = df[df["survey_year"] == YEAR].copy()
    df_year["region"] = region_for_state_fips(df_year["state_fips"])
    df_year = df_year.dropna(subset=["region"])

    train_region_df = df_year[df_year["region"] == TRAIN_REGION]
    train_df, holdout_df = train_test_split(train_region_df, test_size=0.25, random_state=seed)

    train_imp, medians = impute_missing(train_df, feature_cols=feature_cols)
    holdout_imp, _ = impute_missing(holdout_df, medians=medians, feature_cols=feature_cols)
    X_train, y_train = xy(train_imp, feature_cols)
    X_cal, y_cal = xy(holdout_imp, feature_cols)

    xgb = train_xgboost(X_train, y_train, seed=seed)
    scores_cal = lac_scores(xgb, X_cal, y_cal)
    unweighted_threshold = conformal_threshold(scores_cal, alpha=alpha)

    results = {"train_region": TRAIN_REGION, "year": YEAR, "alpha": alpha, "by_region": {}}

    for region in ("Northeast", "Midwest", "South", "West"):
        region_df = holdout_df if region == TRAIN_REGION else df_year[df_year["region"] == region]
        region_imp, _ = impute_missing(region_df, medians=medians, feature_cols=feature_cols)
        X_r, y_r = xy(region_imp, feature_cols)

        unweighted_sets = prediction_sets_from_threshold(xgb, X_r, unweighted_threshold)
        unweighted_coverage = empirical_coverage(y_r, unweighted_sets)
        unweighted_size = mean_set_size(unweighted_sets)

        if region == TRAIN_REGION:
            # weighting calibration against itself is undefined/pointless
            weighted_coverage, weighted_size = unweighted_coverage, unweighted_size
        else:
            weights = estimate_shift_weights(X_cal, X_r, seed=seed)
            weighted_threshold = conformal_threshold(scores_cal, alpha=alpha, weights=weights)
            weighted_sets = prediction_sets_from_threshold(xgb, X_r, weighted_threshold)
            weighted_coverage = empirical_coverage(y_r, weighted_sets)
            weighted_size = mean_set_size(weighted_sets)

        results["by_region"][region] = {
            "n": len(y_r),
            "unweighted_coverage": unweighted_coverage,
            "unweighted_mean_set_size": unweighted_size,
            "weighted_coverage": weighted_coverage,
            "weighted_mean_set_size": weighted_size,
        }
        print(f"{region}: unweighted coverage={unweighted_coverage:.4f} (set size {unweighted_size:.3f})"
              f"  |  weighted coverage={weighted_coverage:.4f} (set size {weighted_size:.3f})")

    Path("reports/weighted_conformal_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to reports/weighted_conformal_results.json")


if __name__ == "__main__":
    main()
