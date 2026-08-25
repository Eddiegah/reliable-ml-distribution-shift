from src.data.geography import STATE_FIPS_TO_REGION, region_for_state_fips


def test_all_51_states_plus_dc_are_mapped_to_exactly_one_region():
    assert len(STATE_FIPS_TO_REGION) == 51
    assert set(STATE_FIPS_TO_REGION.values()) == {"Northeast", "Midwest", "South", "West"}


def test_known_state_fips_codes_map_to_the_correct_region():
    # New York, Texas, California, Illinois, DC — spot-checked against the
    # official Census Bureau reg_div.txt reference table
    assert STATE_FIPS_TO_REGION[36] == "Northeast"
    assert STATE_FIPS_TO_REGION[48] == "South"
    assert STATE_FIPS_TO_REGION[6] == "West"
    assert STATE_FIPS_TO_REGION[17] == "Midwest"
    assert STATE_FIPS_TO_REGION[11] == "South"


def test_region_for_state_fips_is_vectorized():
    regions = region_for_state_fips([36, 48, 6, 17])
    assert list(regions) == ["Northeast", "South", "West", "Midwest"]
