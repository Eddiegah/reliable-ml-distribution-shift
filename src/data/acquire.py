"""Phase 1 — BRFSS acquisition.

Run `python scripts/download_brfss.py` to fetch the four years this study
uses (see brfss_schema.py for why). It downloads straight from CDC's public
annual-data pages — no manual steps or authentication required.
"""

from pathlib import Path

import pandas as pd


def load_year(year: int, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load one year's full raw file (all ~300+ columns). Only used for
    one-off inspection — the real pipeline uses preprocess.extract_year,
    which reads in chunks and keeps just the columns this study needs."""
    path = Path(raw_dir) / f"{year}.xpt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/download_brfss.py {year}` first."
        )
    return pd.read_sas(path, format="xport", encoding="latin-1")


def check_years_available(years: list[int], raw_dir: str = "data/raw") -> dict[int, bool]:
    return {year: (Path(raw_dir) / f"{year}.xpt").exists() for year in years}
