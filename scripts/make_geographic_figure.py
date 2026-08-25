"""Reads reports/geographic_shift_results.json and plots AUROC, calibration
error, and conformal coverage by region. No retraining."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

FIG_DIR = Path("reports/figures")
REGION_ORDER = ["Northeast", "Midwest", "South", "West"]


def main():
    results = json.loads(Path("reports/geographic_shift_results.json").read_text())
    train_region = results["train_region"]
    xgb = results["by_model"]["xgboost"]
    uncertainty = results["uncertainty_calibrated_on_train_region_holdout"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    aurocs, eces, coverages = [], [], []
    for region in REGION_ORDER:
        if region == train_region:
            aurocs.append(xgb["in_region_holdout"]["auroc"])
        else:
            aurocs.append(xgb["out_of_region"][region]["report"]["auroc"])
        eces.append(uncertainty[region]["calibrated_report"]["ece"])
        coverages.append(uncertainty[region]["conformal_coverage"])

    colors = ["#1f77b4" if r == train_region else "#d62728" for r in REGION_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(REGION_ORDER, aurocs, color=colors)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("AUROC")

    axes[1].bar(REGION_ORDER, eces, color=colors)
    axes[1].set_title("Calibration error (ECE)")

    axes[2].bar(REGION_ORDER, coverages, color=colors)
    axes[2].axhline(0.9, linestyle=":", color="black", linewidth=1, label="target (90%)")
    axes[2].set_ylim(0.7, 1.0)
    axes[2].set_title("Conformal coverage")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"XGBoost trained on {train_region} (2023) — evaluated by region "
                 "(blue = in-region, red = out-of-region)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geographic_shift.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'geographic_shift.png'}")


if __name__ == "__main__":
    main()
