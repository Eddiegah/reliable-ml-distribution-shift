"""Phase 1 — BRFSS extraction, recoding, and temporal split.

Column names and recoding rules are documented in brfss_schema.py, verified
against the real downloaded files rather than assumed from memory.
"""

import pandas as pd

from src.data.brfss_schema import FEATURE_COLUMNS, TARGET_COLUMN

CHUNKSIZE = 100_000

# columns present in the clean dataset that are never model features:
# survey_year/diabetes are split/target metadata, state_fips is geographic
# metadata for the extension in src/data/geography.py. Every script that
# computes feature_cols excludes these explicitly and reuses this constant,
# so adding future metadata columns can't silently change what "features"
# means for an already-reported experiment.
NON_FEATURE_COLUMNS = ("survey_year", "diabetes", "state_fips")


def extract_year(year: int, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Read one year's raw .xpt in chunks, keeping only the columns this
    study needs (BRFSS files carry 300+ columns; loading them all at once
    for every year would be several GB in memory simultaneously)."""
    target_col = TARGET_COLUMN[year]
    raw_cols = {name: mapping[year] for name, mapping in FEATURE_COLUMNS.items()}
    wanted_raw = [target_col] + list(raw_cols.values())

    it = pd.read_sas(f"{raw_dir}/{year}.xpt", format="xport", encoding="latin-1",
                      chunksize=CHUNKSIZE, iterator=True)
    chunks = [chunk[wanted_raw] for chunk in it]
    df = pd.concat(chunks, ignore_index=True)

    df = df.rename(columns={target_col: "diabetes_raw", **{v: k for k, v in raw_cols.items()}})
    df["survey_year"] = year
    return df


def recode_target(raw: pd.Series) -> pd.Series:
    """1 = told has diabetes -> 1; 3 = told no -> 0.
    Excluded as ambiguous/non-binary: 2 (gestational-only), 4 (pre-diabetes/
    borderline), 7 (don't know), 9 (refused)."""
    return raw.map({1: 1, 3: 0})


def recode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # "_RF" calculated risk-factor convention: 1=No, 2=Yes (verified against
    # raw BPHIGH4 via crosstab — see brfss_schema.py docstring)
    for col in ("high_bp", "high_chol", "hvy_alcohol"):
        out[col] = df[col].map({1: 0, 2: 1})

    # raw-question convention: 1=Yes, 2=No
    for col in ("smoker", "stroke", "diff_walking", "heart_disease", "phys_activity"):
        out[col] = df[col].map({1: 1, 2: 0})

    out["bmi"] = df["bmi"] / 100.0

    out["gen_health"] = df["gen_health"].where(df["gen_health"].isin([1, 2, 3, 4, 5]))

    for col in ("mental_health_days", "physical_health_days"):
        days = df[col].replace({88: 0})
        out[col] = days.where(days.between(0, 30))

    out["sex_female"] = df["sex"].map({1: 0, 2: 1})
    out["age_group"] = df["age_group"].where(df["age_group"].between(1, 13))
    out["education"] = df["education"].where(df["education"].between(1, 6))

    out["survey_year"] = df["survey_year"]
    out["state_fips"] = df["state_fips"]  # metadata only, see brfss_schema.py
    return out


def build_dataset(years: list[int], raw_dir: str = "data/raw") -> pd.DataFrame:
    """Extract, recode, and concatenate the given years into one clean frame
    with a binary `diabetes` target and no BRFSS sentinel codes remaining."""
    frames = []
    for year in years:
        raw = extract_year(year, raw_dir)
        clean = recode_features(raw)
        clean["diabetes"] = recode_target(raw["diabetes_raw"])
        frames.append(clean)
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["diabetes"])


def impute_missing(
    df: pd.DataFrame, medians: dict | None = None, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Median imputation. Fit medians on the training split only, then reuse
    the same values for val/adapt/test to avoid leaking their distributions
    into the fill values."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    if medians is None:
        medians = {c: float(df[c].median()) for c in feature_cols}
    out = df.copy()
    for c in feature_cols:
        out[c] = out[c].fillna(medians[c])
    return out, medians


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
        splits["test"] = df[test_mask & ~adapt_mask]
    else:
        splits["test"] = df[test_mask]
    return splits
