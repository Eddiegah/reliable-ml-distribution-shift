import numpy as np
import pandas as pd

from src.data.preprocess import impute_missing, recode_features, recode_target, temporal_split


def test_recode_target_maps_yes_no_and_drops_ambiguous_codes():
    raw = pd.Series([1, 3, 2, 4, 7, 9])
    recoded = recode_target(raw)
    assert list(recoded) == [1, 0, None, None, None, None] or recoded.tolist()[:2] == [1, 0]
    assert recoded.isna().sum() == 4


def test_recode_features_applies_rf_convention_and_raw_convention_correctly():
    df = pd.DataFrame({
        "high_bp": [1, 2],        # _RF convention: 1=No, 2=Yes -> 0, 1
        "high_chol": [1, 2],
        "hvy_alcohol": [1, 2],
        "smoker": [1, 2],         # raw convention: 1=Yes, 2=No -> 1, 0
        "stroke": [1, 2],
        "diff_walking": [1, 2],
        "heart_disease": [1, 2],
        "phys_activity": [1, 2],
        "bmi": [2542, 3010],
        "gen_health": [1, 9],     # 9 is not a valid category -> NaN
        "mental_health_days": [88, 15],   # 88 means "none" -> 0
        "physical_health_days": [5, 99],  # 99 is refused -> NaN
        "sex": [1, 2],
        "age_group": [3, 14],     # 14 is missing/dk -> NaN
        "education": [6, 9],      # 9 is refused -> NaN
        "survey_year": [2017, 2017],
        "state_fips": [9, 6],     # passthrough metadata, not recoded
    })
    out = recode_features(df)

    assert out["high_bp"].tolist() == [0, 1]
    assert out["smoker"].tolist() == [1, 0]
    assert np.isclose(out["bmi"].iloc[0], 25.42)
    assert out["gen_health"].iloc[0] == 1 and pd.isna(out["gen_health"].iloc[1])
    assert out["mental_health_days"].tolist() == [0, 15]
    assert pd.isna(out["physical_health_days"].iloc[1])
    assert out["sex_female"].tolist() == [0, 1]
    assert pd.isna(out["age_group"].iloc[1])
    assert out["state_fips"].tolist() == [9, 6]


def test_impute_missing_fills_with_train_medians_not_test_medians():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "diabetes": [0, 1, 0], "survey_year": [2017] * 3})
    test = pd.DataFrame({"x": [np.nan, 100.0], "diabetes": [0, 1], "survey_year": [2023, 2023]})

    _, medians = impute_missing(train)
    test_imputed, _ = impute_missing(test, medians=medians)

    assert medians["x"] == 2.0
    assert test_imputed["x"].iloc[0] == 2.0  # filled with train's median, not test's


def test_temporal_split_keeps_adaptation_sample_out_of_test():
    df = pd.DataFrame({
        "x": range(6),
        "survey_year": [2017, 2017, 2019, 2019, 2023, 2023],
    })
    splits = temporal_split(
        df,
        train_years=[2017],
        val_years=[2019],
        test_years=[2023],
        adaptation_sample_years=[2023],
    )
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 2
    assert len(splits["adapt"]) == 2
    assert len(splits["test"]) == 0  # all 2023 rows went to the adaptation slice
