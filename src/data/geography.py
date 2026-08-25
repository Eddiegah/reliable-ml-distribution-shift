"""US Census Bureau 4-region grouping, by state FIPS code — for the
geographic-shift extension (proposal Section 3). Verified against the
official source (https://www2.census.gov/geo/docs/maps-data/maps/reg_div.txt)
rather than reconstructed from memory. Territories (e.g. Puerto Rico, FIPS 72)
and Guam etc. are intentionally excluded — they don't fit a 4-region
continental grouping and BRFSS samples them differently."""

import numpy as np

NORTHEAST = [9, 23, 25, 33, 44, 50, 34, 36, 42]
MIDWEST = [17, 18, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46]
SOUTH = [10, 11, 12, 13, 24, 37, 45, 51, 54, 1, 21, 28, 47, 5, 22, 40, 48]
WEST = [4, 8, 16, 30, 32, 35, 49, 56, 2, 6, 15, 41, 53]

STATE_FIPS_TO_REGION = {
    **{fips: "Northeast" for fips in NORTHEAST},
    **{fips: "Midwest" for fips in MIDWEST},
    **{fips: "South" for fips in SOUTH},
    **{fips: "West" for fips in WEST},
}


def region_for_state_fips(state_fips) -> np.ndarray:
    state_fips = np.asarray(state_fips)
    return np.array([STATE_FIPS_TO_REGION.get(int(f), None) for f in state_fips], dtype=object)
