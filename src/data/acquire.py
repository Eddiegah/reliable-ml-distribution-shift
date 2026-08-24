"""Phase 1 — BRFSS acquisition.

BRFSS files are not auto-downloaded here: file naming and format have changed
across years, and the raw files are large (tens of MB per year, zipped).
Download manually per year and place under data/raw/<year>.xpt:

  1. Go to https://www.cdc.gov/brfss/annual_data/annual_<year>.html
     (e.g. annual_2022.html for 2022 data)
  2. Download the "SAS Transport Format" (.xpt) zip for that year
  3. Unzip into data/raw/, rename to <year>.xpt

Confirm the exact years to pull with the mentor (configs/default.yaml -> data.*_years)
before downloading — see Section 11 of the research proposal.
"""

from pathlib import Path

import pandas as pd


def load_year(year: int, raw_dir: str = "data/raw") -> pd.DataFrame:
    path = Path(raw_dir) / f"{year}.xpt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — download it manually first (see module docstring)."
        )
    return pd.read_sas(path, format="xport", encoding="latin-1")


def check_years_available(years: list[int], raw_dir: str = "data/raw") -> dict[int, bool]:
    return {year: (Path(raw_dir) / f"{year}.xpt").exists() for year in years}
