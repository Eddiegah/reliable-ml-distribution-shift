"""Phase 1 — download the four BRFSS years this study uses.

Each URL below was verified by hand (HTTP 200, correct zip contents) before
being hardcoded here — see brfss_schema.py for why these specific years.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.brfss_schema import SUPPORTED_YEARS

URL_TEMPLATE = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"


def download_year(year: int, raw_dir: str = "data/raw") -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{year}.xpt"
    if dest.exists():
        print(f"{year}: already present at {dest}, skipping")
        return dest

    zip_path = raw_dir / f"{year}.zip"
    url = URL_TEMPLATE.format(year=year)
    print(f"{year}: downloading {url}")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        (member,) = zf.namelist()  # exactly one file per BRFSS year zip
        zf.extract(member, raw_dir)
        # BRFSS zips store the filename with a trailing space (e.g. "LLCP2022.XPT ")
        (raw_dir / member).rename(dest)
    zip_path.unlink()
    print(f"{year}: saved to {dest}")
    return dest


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or list(SUPPORTED_YEARS)
    for y in years:
        download_year(y)
