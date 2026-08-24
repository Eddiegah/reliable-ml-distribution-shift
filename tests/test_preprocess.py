import pandas as pd

from src.data.preprocess import clean_missing, concat_years, temporal_split


def test_clean_missing_replaces_brfss_sentinel_codes_with_na():
    df = pd.DataFrame({"x": [1, 7, 9, 4]})
    cleaned = clean_missing(df)
    assert cleaned["x"].isna().sum() == 2
    assert cleaned["x"].iloc[0] == 1


def test_concat_years_tags_each_row_with_its_survey_year():
    frames = {
        2015: pd.DataFrame({"x": [1, 2]}),
        2016: pd.DataFrame({"x": [3, 4]}),
    }
    combined = concat_years(frames)
    assert list(combined["survey_year"]) == [2015, 2015, 2016, 2016]


def test_temporal_split_keeps_adaptation_sample_out_of_test():
    df = pd.DataFrame({
        "x": range(6),
        "survey_year": [2015, 2015, 2018, 2018, 2021, 2021],
    })
    splits = temporal_split(
        df,
        train_years=[2015],
        val_years=[2018],
        test_years=[2021],
        adaptation_sample_years=[2021],
    )
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 2
    assert len(splits["adapt"]) == 2
    assert len(splits["test"]) == 0  # all 2021 rows went to the adaptation slice
