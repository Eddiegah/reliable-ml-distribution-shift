"""Deep-ensemble / MC-dropout uncertainty baseline (Ovadia et al., 2019),
compared against the calibrated-XGBoost results from run_pipeline.py.

Runs on CPU by default (fine for a correctness check at small scale). Once
AMD MI300X access is available, install the ROCm PyTorch build and re-run
with --epochs and --n-members at the scale intended for the paper — this
script does not change, only the hardware and those two flags do.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.preprocess import NON_FEATURE_COLUMNS, impute_missing, temporal_split
from src.evaluation.metrics import performance_report
from src.models.neural import ensemble_predict, get_device, mc_dropout_predict, train_deep_ensemble
from src.uncertainty.failure_detection import failure_detection_auroc
from src.utils.config import load_config
from src.utils.seed import set_seed


def xy(df: pd.DataFrame, feature_cols: list[str]):
    return df[feature_cols].to_numpy(dtype=np.float32), df["diabetes"].to_numpy(dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--tag", default="full", help="label for this run in the saved results file")
    args = parser.parse_args()

    config = load_config()
    set_seed(config["model"]["seed"])
    device = get_device()
    print(f"device: {device}  epochs: {args.epochs}  ensemble members: {args.n_members}")

    df = pd.read_parquet("data/processed/brfss_clean.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    splits = temporal_split(
        df,
        train_years=config["data"]["train_years"],
        val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"],
        adaptation_sample_years=config["data"]["adaptation_sample_years"],
    )
    train_imp, medians = impute_missing(splits["train"], feature_cols=feature_cols)
    val_imp, _ = impute_missing(splits["val"], medians=medians, feature_cols=feature_cols)
    test_imp, _ = impute_missing(splits["test"], medians=medians, feature_cols=feature_cols)

    X_train, y_train = xy(train_imp, feature_cols)
    X_val, y_val = xy(val_imp, feature_cols)
    X_test, y_test = xy(test_imp, feature_cols)

    # standardize — unlike tree models, neural nets are sensitive to feature scale
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train, X_val, X_test = ((X - mean) / std for X in (X_train, X_val, X_test))

    models = train_deep_ensemble(
        X_train, y_train, n_members=args.n_members, seed=config["model"]["seed"],
        epochs=args.epochs, device=device,
    )

    val_mean, _ = ensemble_predict(models, X_val, device=device)
    test_mean, test_per_member = ensemble_predict(models, X_test, device=device)

    val_report = performance_report(y_val, val_mean)
    test_report = performance_report(y_test, test_mean)
    print(f"\n[deep ensemble] val(2019) AUROC={val_report['auroc']:.4f} ECE={val_report['ece']:.4f}"
          f"  |  test(2023) AUROC={test_report['auroc']:.4f} ECE={test_report['ece']:.4f}")

    # ensemble disagreement (std across members) as the uncertainty signal
    ensemble_std = test_per_member.std(axis=0)
    y_pred = (test_mean >= 0.5).astype(int)
    fd_auroc_ensemble = failure_detection_auroc(y_test, y_pred, ensemble_std)
    print(f"[deep ensemble] failure-detection AUROC (ensemble disagreement) = {fd_auroc_ensemble:.4f}")

    # MC-dropout on a single member, for comparison
    _, mc_samples = mc_dropout_predict(models[0], X_test, n_samples=20, device=device)
    mc_std = mc_samples.std(axis=0)
    fd_auroc_mc = failure_detection_auroc(y_test, y_pred, mc_std)
    print(f"[MC-dropout, single net] failure-detection AUROC = {fd_auroc_mc:.4f}")

    results = {
        "device": str(device),
        "epochs": args.epochs,
        "n_members": args.n_members,
        "val_2019": val_report,
        "test_2023": test_report,
        "failure_detection_auroc_ensemble_disagreement": float(fd_auroc_ensemble),
        "failure_detection_auroc_mc_dropout": float(fd_auroc_mc),
    }
    out_path = Path(f"reports/deep_ensemble_results_{args.tag}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")

    # raw predictions, not just summary metrics — without these, a bootstrap
    # CI on any of the numbers above needs a full GPU retrain to reconstruct
    predictions_path = Path(f"reports/deep_ensemble_predictions_{args.tag}.npz")
    np.savez(
        predictions_path,
        y_val=y_val, val_mean=val_mean,
        y_test=y_test, test_mean=test_mean, test_per_member=test_per_member,
        mc_samples=mc_samples,
    )
    print(f"Saved raw predictions to {predictions_path}")


if __name__ == "__main__":
    main()
