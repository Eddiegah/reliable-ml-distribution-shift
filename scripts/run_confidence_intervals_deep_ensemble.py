"""Bootstrap confidence intervals for Section 4.5 (deep ensemble / MC-dropout),
the one section that didn't have them - the earlier GPU run only persisted
summary metrics, not raw predictions. Uses the raw predictions from the
rerun (reports/deep_ensemble_predictions_full.npz) plus a fresh XGBoost fit
on the identical temporal split, so the ensemble-vs-XGBoost AUROC gap gets
a real paired CI on the same 2023 test rows."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.preprocess import NON_FEATURE_COLUMNS, impute_missing, temporal_split
from src.evaluation.bootstrap import bootstrap_metric, bootstrap_paired_diff, fast_auroc
from src.evaluation.metrics import expected_calibration_error
from src.models.baselines import train_xgboost
from src.utils.config import load_config
from src.utils.seed import set_seed

N_BOOT = 300


def fd_auroc_metric(y_true, packed):
    y_pred, uncertainty = packed[:, 0].astype(int), packed[:, 1]
    is_error = (y_true != y_pred).astype(int)
    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return 0.5
    return fast_auroc(is_error, uncertainty)


def main():
    data = np.load("reports/deep_ensemble_predictions_full.npz")
    y_test, test_mean, test_per_member, mc_samples = (
        data["y_test"], data["test_mean"], data["test_per_member"], data["mc_samples"]
    )
    results = {}

    ci_auroc = bootstrap_metric(y_test, test_mean, fast_auroc, n_boot=N_BOOT, seed=42)
    ci_ece = bootstrap_metric(y_test, test_mean, expected_calibration_error, n_boot=N_BOOT, seed=42)
    print(f"[4.5] Deep ensemble AUROC(test)={ci_auroc['point']:.4f} "
          f"[{ci_auroc['ci_low']:.4f}, {ci_auroc['ci_high']:.4f}]  "
          f"ECE(raw)={ci_ece['point']:.4f} [{ci_ece['ci_low']:.4f}, {ci_ece['ci_high']:.4f}]")

    y_pred = (test_mean >= 0.5).astype(int)
    ensemble_std = test_per_member.std(axis=0)
    mc_std = mc_samples.std(axis=0)
    packed_ens = np.column_stack([y_pred, ensemble_std])
    packed_mc = np.column_stack([y_pred, mc_std])

    ci_fd_ens = bootstrap_metric(y_test, packed_ens, fd_auroc_metric, n_boot=N_BOOT, seed=42)
    ci_fd_mc = bootstrap_metric(y_test, packed_mc, fd_auroc_metric, n_boot=N_BOOT, seed=42)
    print(f"[4.5] Failure-detection AUROC: ensemble disagreement={ci_fd_ens['point']:.4f} "
          f"[{ci_fd_ens['ci_low']:.4f}, {ci_fd_ens['ci_high']:.4f}]  "
          f"MC-dropout={ci_fd_mc['point']:.4f} [{ci_fd_mc['ci_low']:.4f}, {ci_fd_mc['ci_high']:.4f}]")

    ci_fd_gap = bootstrap_paired_diff(y_test, packed_ens, packed_mc, fd_auroc_metric, n_boot=N_BOOT, seed=42)
    print(f"[4.5] Failure-detection gap (ensemble - MC-dropout)={ci_fd_gap['point']:.4f} "
          f"[{ci_fd_gap['ci_low']:.4f}, {ci_fd_gap['ci_high']:.4f}]")

    # fresh XGBoost on the identical temporal split, for a real paired AUROC-gap CI
    config = load_config()
    seed = config["model"]["seed"]
    set_seed(seed)
    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    splits = temporal_split(
        df, train_years=config["data"]["train_years"], val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"], adaptation_sample_years=config["data"]["adaptation_sample_years"],
    )
    train_imp, medians = impute_missing(splits["train"], feature_cols=feature_cols)
    test_imp, _ = impute_missing(splits["test"], medians=medians, feature_cols=feature_cols)
    X_train = train_imp[feature_cols].to_numpy(dtype=float)
    y_train = train_imp["diabetes"].to_numpy(dtype=int)
    X_test = test_imp[feature_cols].to_numpy(dtype=float)
    y_test_xgb = test_imp["diabetes"].to_numpy(dtype=int)

    assert len(y_test_xgb) == len(y_test), "row-count mismatch between XGBoost split and deep-ensemble npz"
    assert np.array_equal(y_test_xgb, y_test), "row-order mismatch between XGBoost split and deep-ensemble npz"

    xgb = train_xgboost(X_train, y_train, seed=seed)
    xgb_test_probs = xgb.predict_proba(X_test)[:, 1]

    ci_gap = bootstrap_paired_diff(y_test, test_mean.astype(float), xgb_test_probs, fast_auroc, n_boot=N_BOOT, seed=42)
    print(f"[4.5] AUROC gap (deep ensemble - XGBoost raw) = {ci_gap['point']:.4f} "
          f"[{ci_gap['ci_low']:.4f}, {ci_gap['ci_high']:.4f}]")

    results = {
        "auroc_test": ci_auroc,
        "ece_raw_test": ci_ece,
        "failure_detection_auroc_ensemble_disagreement": ci_fd_ens,
        "failure_detection_auroc_mc_dropout": ci_fd_mc,
        "failure_detection_gap_ensemble_minus_mc": ci_fd_gap,
        "auroc_gap_ensemble_minus_xgboost": ci_gap,
    }
    Path("reports/confidence_intervals_deep_ensemble.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to reports/confidence_intervals_deep_ensemble.json")


if __name__ == "__main__":
    main()
