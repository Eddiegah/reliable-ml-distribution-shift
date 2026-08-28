# Reliable ML Under Distribution Shift
### Uncertainty-Aware Health Risk Prediction

![CI](https://github.com/Eddiegah/reliable-ml-distribution-shift/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-paper--ready-brightgreen)

**Does a model know when it might be wrong?**

Machine-learning models can look excellent in development and still fail quietly in deployment, not by crashing, but by staying confidently wrong once the data they see no longer resembles the data they were trained on. In healthcare that gap is dangerous: patient populations, practices, and data collection drift across time, geography, and demographics, and a model that cannot signal when it is out of its depth is a model you cannot trust.

This project trains diabetes-risk classifiers on the CDC's [Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/) (BRFSS) and evaluates them under three real, non-synthetic distribution shifts, temporal, demographic, and geographic, asking a sharper question than standard evaluation does: when the model is confident, is that confidence earned? Calibrated probabilities and conformal prediction are tested as the mechanism for flagging unreliable predictions, and weighted conformal prediction is tested as the remedy once shift is detected.

## Paper

| | |
|---|---|
| **Final PDF** | [`docs/paper_latex/main.pdf`](docs/paper_latex/main.pdf) |
| **Overleaf package** (upload directly to edit, e.g. to add a co-author) | [`docs/paper_latex/Gah_Reliable_ML_Paper_Overleaf.zip`](docs/paper_latex/Gah_Reliable_ML_Paper_Overleaf.zip) |
| **Markdown source** (readable on GitHub) | [`docs/paper_draft.md`](docs/paper_draft.md) |
| **Word version** | [`docs/paper_draft.docx`](docs/paper_draft.docx) |
| **Original research proposal** | [`docs/PROPOSAL.md`](docs/PROPOSAL.md) · [`docs/proposal.docx`](docs/proposal.docx) |

Target venue: **ML4H 2026** (Machine Learning for Health Symposium), submission deadline September 10, 2026.

## Key findings

![Risk-coverage curve](reports/figures/risk_coverage_curve.png)

| Analysis | Headline result |
|---|---|
| Temporal shift (2017 → 2023) | AUROC drop of 0.008 [95% CI 0.006, 0.010]: mild but statistically real |
| Subgroup fairness | AUROC 0.836 (ages 18–44) vs. 0.750 (65+): a gap the aggregate number hides entirely |
| Geographic shift (single year, region vs. region) | Conformal coverage drop roughly 9x larger than the full 6-year temporal shift |
| Weighted conformal prediction | Recovers about half the coverage lost to geographic shift, at a small but real cost where no shift needed correcting |
| Deep ensemble vs. XGBoost (AMD Instinct MI300X) | Statistically tied on AUROC; the uncertainty signal replicates across two structurally different model families |

Every number above carries a 95% bootstrap confidence interval rather than being reported as a bare point estimate. Full method, all eight figures, and discussion are in the [paper](docs/paper_latex/main.pdf).

## Research question

> When a health-risk prediction model encounters distribution shift, can uncertainty estimation reliably identify predictions that are likely to be incorrect, and can a known remedy restore reliable coverage once it degrades?

## Related work

This targets a specific, underexplored corner of ML reliability: naturally occurring distribution shift in tabular healthcare survey data, rather than synthetic corruption on image or text benchmarks. Grounded in:

- Ovadia et al. (2019), Lakshminarayanan et al. (2017), Gal & Ghahramani (2016): uncertainty estimates, including deep ensembles and MC-dropout (both implemented here), degrade under dataset shift
- Guo et al. (2017); Niculescu-Mizil & Caruana (2005): post-hoc calibration is effective in-distribution but less studied under shift
- Vovk, Gammerman & Shafer (2005); Tibshirani et al. (2019): conformal prediction's coverage guarantees break under covariate shift
- Koh et al. (2021): WILDS, the standard shift benchmark, is mostly image and text
- Malinin et al. (2021): uncertainty specifically for gradient-boosted trees, this project's primary model family
- Guo et al. (2022, 2023): the closest prior design, splitting MIMIC-IV into year groups to study temporal shift directly, though on hospital EHR data and centered on adaptation rather than uncertainty-as-failure-detector

All 11 references are verified against primary sources. Full list with arXiv/DOI links in the [paper's references](docs/paper_draft.md#references).

## Data and decisions

- **Target:** BRFSS diabetes indicator (`DIABETE3`/`DIABETE4`, renamed partway through the study window; confirmed to share identical coding)
- **Years:** 2017 (train) → 2019 (validation) → 2021 (adaptation sample) → 2023 (final test). Restricted to odd years: 2018 and 2022 were checked against the real files and completely lack the blood-pressure/cholesterol survey module that year, which would have confounded "distribution shift" with "missing feature." Full rationale in [`src/data/brfss_schema.py`](src/data/brfss_schema.py).
- **Compute:** classical baselines ran on local CPU. The deep-learning uncertainty extension ran on an AMD Instinct MI300X, provided by AMD via Exea Labs.

## Acknowledgments

This research was conducted with mentorship from **Avneh Singh Bhatia** at **Exea Labs**, who arranged access to an **AMD Instinct MI300X** accelerator, provided by **AMD**, for the deep-learning experiments in the paper.

That access is why the deep-ensemble section is a completed result rather than a CPU-only correctness check: the MI300X ran the full-scale 10-member, 50-epoch ensemble in minutes on a pre-configured ROCm/PyTorch environment with no setup friction. That let this project test its central finding, that uncertainty survives distribution shift, on a second, structurally different model family rather than resting on gradient-boosted trees alone. Thanks to AMD and Exea Labs for making that possible.

## Quickstart

```bash
python scripts/download_brfss.py    # ~450 MB, four BRFSS years, public CDC data
python scripts/build_dataset.py     # extract + clean -> data/processed/brfss_clean.parquet
python scripts/run_pipeline.py      # baselines, shift eval, calibration, conformal, adaptation
python scripts/run_subgroup_analysis.py   # fairness by sex/age
python scripts/run_geographic_shift.py    # shift by US Census region
python scripts/run_weighted_conformal.py  # does reweighting recover the lost coverage?
python scripts/run_deep_ensemble.py --epochs 50 --n-members 10  # needs a GPU for reasonable runtime
python scripts/run_confidence_intervals.py                 # bootstrap CIs, classical baselines
python scripts/run_confidence_intervals_deep_ensemble.py   # bootstrap CIs, deep ensemble / MC-dropout
```

## Setup

Requires Python 3.10+ (type hints use the `X | Y` syntax).

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Development & tests

All model-agnostic modules (metrics, calibration, conformal prediction, baseline training, adaptation, temporal splitting, bootstrap CIs) are covered by a passing test suite. CI runs it on every push.

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -v
```

## Repo layout

```
configs/          run configuration (target variable, years, seeds)
scripts/          runnable entry points: download data, build dataset, run each analysis
src/data/         acquisition, cleaning, temporal split
src/models/       baseline training (Logistic Regression, Random Forest, XGBoost) and neural models
src/uncertainty/  calibration, split and weighted conformal prediction
src/adaptation/   recalibration on a held-out target-period sample
src/evaluation/   performance/calibration metrics, subgroup and bootstrap analysis
tests/            pytest suite for every model-agnostic module
data/raw/         downloaded BRFSS files (gitignored)
data/processed/   cleaned/split datasets (gitignored)
reports/          figures and result artifacts
docs/             proposal, paper (Markdown, Word, and LaTeX/Overleaf), literature notes
```

## Contributors

| | |
|---|---|
| **Edmund Eric Gah** | Researcher — [gahedmund146@gmail.com](mailto:gahedmund146@gmail.com) |
| **Avneh Singh Bhatia** | Mentor, Exea Labs |

## License

Released under the [MIT License](LICENSE). See [`CITATION.cff`](CITATION.cff) for citation.
