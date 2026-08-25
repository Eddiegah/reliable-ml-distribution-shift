"""Phase 1 — build the clean, multi-year dataset from raw BRFSS files.
Run scripts/download_brfss.py first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.brfss_schema import SUPPORTED_YEARS
from src.data.preprocess import build_dataset

if __name__ == "__main__":
    df = build_dataset(list(SUPPORTED_YEARS))

    print(f"shape={df.shape}")
    print(df["survey_year"].value_counts().sort_index())
    print("\ndiabetes rate by year:")
    print(df.groupby("survey_year")["diabetes"].mean())
    print("\nmissingness:")
    print(df.isna().mean().sort_values(ascending=False))

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/processed/brfss_clean.parquet", index=False)
    print("\nsaved to data/processed/brfss_clean.parquet")
