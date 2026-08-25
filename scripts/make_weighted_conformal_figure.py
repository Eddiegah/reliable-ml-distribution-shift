"""Reads reports/weighted_conformal_results.json and plots unweighted vs
weighted conformal coverage by region. No retraining."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("reports/figures")
REGION_ORDER = ["Northeast", "Midwest", "South", "West"]


def main():
    results = json.loads(Path("reports/weighted_conformal_results.json").read_text())
    by_region = results["by_region"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    unweighted = [by_region[r]["unweighted_coverage"] for r in REGION_ORDER]
    weighted = [by_region[r]["weighted_coverage"] for r in REGION_ORDER]

    x = np.arange(len(REGION_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, unweighted, width, label="unweighted (split conformal)", color="#d62728")
    ax.bar(x + width / 2, weighted, width, label="weighted (Tibshirani et al. 2019)", color="#1f77b4")
    ax.axhline(1 - results["alpha"], linestyle=":", color="black", linewidth=1, label="target")
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.8, 0.95)
    ax.set_title(f"Conformal coverage: unweighted vs. weighted (trained on {results['train_region']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weighted_conformal.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'weighted_conformal.png'}")


if __name__ == "__main__":
    main()
