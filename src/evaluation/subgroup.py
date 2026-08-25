"""Extension (Section 3 of the proposal) — subgroup fairness on the shifted
test set. Checks whether the mild aggregate degradation found in
run_pipeline.py hides a larger degradation within a specific subgroup."""

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import expected_calibration_error


def bin_age_group(age_group: np.ndarray) -> np.ndarray:
    """_AGEG5YR is 13 ordinal 5-year bins (1 = 18-24 ... 13 = 80+). Collapse
    to three bands wide enough to have solid sample sizes in every group."""
    age_group = np.asarray(age_group)
    bands = np.where(age_group <= 5, "18-44", np.where(age_group <= 9, "45-64", "65+"))
    return bands


def subgroup_performance(y_true, y_prob, groups) -> pd.DataFrame:
    """Per-group AUROC, ECE, base rate, and sample size."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "group": groups})
    rows = []
    for group_value, sub in df.groupby("group"):
        rows.append({
            "group": group_value,
            "n": len(sub),
            "base_rate": sub["y_true"].mean(),
            "auroc": roc_auc_score(sub["y_true"], sub["y_prob"]) if sub["y_true"].nunique() > 1 else float("nan"),
            "ece": expected_calibration_error(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy()),
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def fairness_gaps(y_true, y_pred, groups) -> dict:
    """Standard Fairlearn gap metrics at a fixed decision threshold."""
    return {
        "demographic_parity_difference": float(demographic_parity_difference(y_true, y_pred, sensitive_features=groups)),
        "equalized_odds_difference": float(equalized_odds_difference(y_true, y_pred, sensitive_features=groups)),
    }
