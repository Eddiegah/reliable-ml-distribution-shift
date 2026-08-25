"""Extension — does the mild aggregate shift (run_pipeline.py) hide a
larger degradation within a demographic subgroup? Checks XGBoost, calibrated
on val(2019), evaluated on the shifted test(2023) set, broken out by sex and
age band."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocess import NON_FEATURE_COLUMNS, impute_missing, temporal_split
from src.evaluation.subgroup import bin_age_group, fairness_gaps, subgroup_performance
from src.models.baselines import train_xgboost
from src.uncertainty.calibration import calibrate, predict_proba_positive
from src.utils.config import load_config
from src.utils.seed import set_seed


def xy(df, feature_cols):
    return df[feature_cols].to_numpy(dtype=float), df["diabetes"].to_numpy(dtype=int)


def main():
    config = load_config()
    set_seed(config["model"]["seed"])

    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    splits = temporal_split(
        df,
        train_years=config["data"]["train_years"],
        val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"],
        adaptation_sample_years=config["data"]["adaptation_sample_years"],
    )
    train_imp, medians = impute_missing(splits["train"], feature_cols=feature_cols)
    val_imp, _ = impute_missing(splits["val"], medians=medians, feature_cols=feature_cols)
    test_imp, _ = impute_missing(splits["test"], medians=medians, feature_cols=feature_cols)

    X_train, y_train = xy(train_imp, feature_cols)
    X_val, y_val = xy(val_imp, feature_cols)
    X_test, y_test = xy(test_imp, feature_cols)

    xgb = train_xgboost(X_train, y_train, seed=config["model"]["seed"])
    calibrated = calibrate(xgb, X_val, y_val, method=config["calibration"]["method"])
    test_probs = predict_proba_positive(calibrated, X_test)
    test_pred = (test_probs >= 0.5).astype(int)

    results = {}

    sex_labels = test_imp["sex_female"].map({0: "male", 1: "female"}).to_numpy()
    by_sex = subgroup_performance(y_test, test_probs, sex_labels)
    print("\nBy sex (test 2023):")
    print(by_sex.to_string(index=False))
    results["by_sex"] = by_sex.to_dict(orient="records")
    results["fairness_gaps_sex"] = fairness_gaps(y_test, test_pred, sex_labels)
    print(f"Fairness gaps (sex): {results['fairness_gaps_sex']}")

    age_bands = bin_age_group(test_imp["age_group"].to_numpy())
    by_age = subgroup_performance(y_test, test_probs, age_bands)
    print("\nBy age band (test 2023):")
    print(by_age.to_string(index=False))
    results["by_age_band"] = by_age.to_dict(orient="records")
    results["fairness_gaps_age_band"] = fairness_gaps(y_test, test_pred, age_bands)
    print(f"Fairness gaps (age band): {results['fairness_gaps_age_band']}")

    Path("reports/subgroup_fairness.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to reports/subgroup_fairness.json")


if __name__ == "__main__":
    main()
