"""Phase 1 — cleaning and temporal split.

BRFSS commonly codes "don't know" / "refused" as 7, 9, 77, or 99 depending on the
question's digit width — verify the exact codes for the chosen target/features
against that year's codebook before relying on the defaults here.
"""

import pandas as pd

DEFAULT_MISSING_CODES = [7, 9, 77, 99, 777, 999]


def clean_missing(df: pd.DataFrame, missing_codes: list[int] = DEFAULT_MISSING_CODES) -> pd.DataFrame:
    return df.replace(missing_codes, pd.NA)


def concat_years(year_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for year, df in year_frames.items():
        df = df.copy()
        df["survey_year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def temporal_split(
    df: pd.DataFrame,
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    adaptation_sample_years: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    test_mask = df["survey_year"].isin(test_years)
    splits = {
        "train": df[df["survey_year"].isin(train_years)],
        "val": df[df["survey_year"].isin(val_years)],
    }
    if adaptation_sample_years:
        adapt_mask = df["survey_year"].isin(adaptation_sample_years)
        splits["adapt"] = df[adapt_mask]
        # keep the adaptation slice out of final test evaluation
        splits["test"] = df[test_mask & ~adapt_mask]
    else:
        splits["test"] = df[test_mask]
    return splits
