"""Reads reports/subgroup_fairness.json (from run_subgroup_analysis.py) and
plots AUROC by subgroup. No retraining — just visualizes saved results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

FIG_DIR = Path("reports/figures")


def main():
    results = json.loads(Path("reports/subgroup_fairness.json").read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    sex_rows = results["by_sex"]
    axes[0].bar([r["group"] for r in sex_rows], [r["auroc"] for r in sex_rows], color=["#d62728", "#1f77b4"])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("By sex")

    age_rows = sorted(results["by_age_band"], key=lambda r: r["group"])
    axes[1].bar([r["group"] for r in age_rows], [r["auroc"] for r in age_rows], color="#2ca02c")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("By age band")

    fig.suptitle("XGBoost (calibrated) AUROC by subgroup — shifted test set (2023)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "subgroup_auroc.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'subgroup_auroc.png'}")


if __name__ == "__main__":
    main()
