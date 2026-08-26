"""Reads reports/deep_ensemble_results_full.json (the real GPU run) and
reports/phase2-5_results.json (XGBoost) and plots them side by side.
No retraining."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

FIG_DIR = Path("reports/figures")


def main():
    ensemble = json.loads(Path("reports/deep_ensemble_results_full.json").read_text())
    xgb_all = json.loads(Path("reports/phase2-5_results.json").read_text())["xgboost"]

    models = ["Deep Ensemble\n(10 members)", "XGBoost\n(raw)"]
    test_auroc = [ensemble["test_2023"]["auroc"], xgb_all["test_2023_shifted_raw"]["auroc"]]
    test_ece = [ensemble["test_2023"]["ece"], xgb_all["test_2023_shifted_raw"]["ece"]]
    fd_auroc = [
        ensemble["failure_detection_auroc_ensemble_disagreement"],
        ensemble["failure_detection_auroc_mc_dropout"],
        xgb_all["failure_detection_auroc_test_2023"],
    ]
    fd_labels = ["Deep ensemble\n(disagreement)", "MC-dropout", "XGBoost\n(calibrated)"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].bar(models, test_auroc, color=["#1f77b4", "#d62728"])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("AUROC (test 2023)")

    axes[1].bar(models, test_ece, color=["#1f77b4", "#d62728"])
    axes[1].set_title("ECE, raw / uncalibrated (test 2023)")

    axes[2].bar(fd_labels, fd_auroc, color=["#1f77b4", "#9467bd", "#d62728"])
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("Failure-detection AUROC")

    fig.suptitle("Deep ensemble / MC-dropout (AMD MI300X) vs. XGBoost — test 2023")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deep_ensemble_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'deep_ensemble_comparison.png'}")


if __name__ == "__main__":
    main()
