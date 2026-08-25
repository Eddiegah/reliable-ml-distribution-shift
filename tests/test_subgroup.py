import numpy as np

from src.evaluation.subgroup import bin_age_group, fairness_gaps, subgroup_performance


def test_bin_age_group_collapses_13_categories_into_three_bands():
    age_group = np.array([1, 5, 6, 9, 10, 13])
    bands = bin_age_group(age_group)
    assert list(bands) == ["18-44", "18-44", "45-64", "45-64", "65+", "65+"]


def test_subgroup_performance_reports_one_row_per_group():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.55, 0.45])
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    result = subgroup_performance(y_true, y_prob, groups)
    assert set(result["group"]) == {"a", "b"}
    assert result.set_index("group").loc["a", "n"] == 4


def test_fairness_gaps_returns_zero_when_groups_behave_identically():
    y_true = np.array([0, 1] * 10)
    y_pred = np.array([0, 1] * 10)
    groups = np.array((["a"] * 10) + (["b"] * 10))
    gaps = fairness_gaps(y_true, y_pred, groups)
    assert gaps["demographic_parity_difference"] == 0.0
    assert gaps["equalized_odds_difference"] == 0.0
