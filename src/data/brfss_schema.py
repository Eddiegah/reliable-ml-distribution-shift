"""BRFSS column names for the same underlying question shift across survey years.
Every mapping below was verified against the real downloaded files for
2017/2019/2021/2023 (see notebooks/phase1_data_exploration notes), not guessed
from the codebook alone — in particular:

  - DIABETE3 (2017) and DIABETE4 (2019+) share identical response coding
    (confirmed via matching value-count distributions).
  - _RFHYPE5/_RFHYPE6, _RFCHOL1/2/3, _RFDRHV5/7/8, SEX/SEXVAR are the same
    underlying "_RF" calculated risk-factor variable, renamed but not
    recoded (confirmed _RFHYPE5 against raw BPHIGH4 via crosstab: BPHIGH4==1
    maps entirely to _RFHYPE5==2, i.e. the BRFSS "_RF" convention is 1=No, 2=Yes).
  - 2018 and 2022 were dropped entirely: BRFSS's hypertension/cholesterol
    module is asked only in odd years and is completely absent in those two
    years' files. Using an even year as the shifted/test period would have
    made "no blood pressure feature" indistinguishable from "distribution
    shift," so the study uses 2017 / 2019 / 2021 / 2023 instead.
  - Fruit/vegetable intake (_FRTLT1A/_VEGLT1A) and income (INCOME2/INCOME3)
    were dropped from the feature set: the fruit/veg module was removed from
    the core survey by 2023, and the income question's category definitions
    changed in 2021 — both are real questionnaire redesigns, not something
    we want the "distribution shift" analysis to silently absorb.
"""

TARGET_COLUMN = {2017: "DIABETE3", 2019: "DIABETE4", 2021: "DIABETE4", 2023: "DIABETE4"}

FEATURE_COLUMNS = {
    "high_bp": {2017: "_RFHYPE5", 2019: "_RFHYPE5", 2021: "_RFHYPE6", 2023: "_RFHYPE6"},
    "high_chol": {2017: "_RFCHOL1", 2019: "_RFCHOL2", 2021: "_RFCHOL3", 2023: "_RFCHOL3"},
    "bmi": {y: "_BMI5" for y in (2017, 2019, 2021, 2023)},
    "smoker": {y: "SMOKE100" for y in (2017, 2019, 2021, 2023)},
    "stroke": {y: "CVDSTRK3" for y in (2017, 2019, 2021, 2023)},
    "heart_disease": {y: "_MICHD" for y in (2017, 2019, 2021, 2023)},
    "phys_activity": {y: "_TOTINDA" for y in (2017, 2019, 2021, 2023)},
    "hvy_alcohol": {2017: "_RFDRHV5", 2019: "_RFDRHV7", 2021: "_RFDRHV7", 2023: "_RFDRHV8"},
    "gen_health": {y: "GENHLTH" for y in (2017, 2019, 2021, 2023)},
    "mental_health_days": {y: "MENTHLTH" for y in (2017, 2019, 2021, 2023)},
    "physical_health_days": {y: "PHYSHLTH" for y in (2017, 2019, 2021, 2023)},
    "diff_walking": {y: "DIFFWALK" for y in (2017, 2019, 2021, 2023)},
    "sex": {2017: "SEX", 2019: "SEXVAR", 2021: "SEXVAR", 2023: "SEXVAR"},
    "age_group": {y: "_AGEG5YR" for y in (2017, 2019, 2021, 2023)},
    "education": {y: "EDUCA" for y in (2017, 2019, 2021, 2023)},
    # kept as metadata for the geographic-shift extension (src/data/geography.py),
    # NOT a model feature — every feature_cols computation elsewhere excludes it
    # explicitly, same as survey_year/diabetes, so this addition does not change
    # any previously reported temporal/subgroup result.
    "state_fips": {y: "_STATE" for y in (2017, 2019, 2021, 2023)},
}

SUPPORTED_YEARS = (2017, 2019, 2021, 2023)

# recode rules applied after extraction — see preprocess.recode_features/recode_target
YES_NO_RF = {1: 0, 2: 1}          # "_RF" convention: 1=No, 2=Yes -> 0/1
YES_NO_RAW = {1: 1, 2: 0}          # raw question convention: 1=Yes, 2=No -> 0/1
DAYS_WITH_NONE_CODE = {88: 0}      # 88 means "none" on the 1-30 day-count questions
