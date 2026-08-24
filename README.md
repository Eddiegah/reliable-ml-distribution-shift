# Reliable ML Under Distribution Shift — Uncertainty-Aware Health Risk Prediction

Researcher: Edmund Eric Gah
Full proposal: [`docs/proposal.docx`](docs/proposal.docx)

Does a health-risk model know when it might be wrong once the data distribution
shifts — and can lightweight recalibration recover what's lost? This repo trains
BRFSS-based diabetes-risk classifiers, evaluates them under real temporal
distribution shift (train on earlier survey years, test on later ones), and
tests whether calibrated probabilities and conformal prediction correctly flag
unreliable predictions.

## Status: open items before Phase 1

`configs/default.yaml` has several fields marked `TBD` — confirm these with the
lab before running the pipeline (see Section 11 of the proposal):

- [ ] Target variable (which BRFSS diabetes field, and which years it's coded consistently)
- [ ] Train / validation / test / adaptation-sample years
- [ ] Total time horizon for the engagement (scales the phase estimates below)

## Setup

Requires Python 3.10+ (type hints use the `X | Y` syntax).

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Phases

| Phase | Focus | Entry point |
|---|---|---|
| 0 | Literature review | `docs/` — add notes here |
| 1 | Data acquisition & preprocessing | `src/data/acquire.py`, `src/data/preprocess.py` |
| 2 | Baseline models (Logistic Regression, XGBoost) | `src/models/baselines.py` |
| 3 | Distribution shift evaluation | `src/evaluation/metrics.py` (`performance_degradation`) |
| 4 | Uncertainty quantification (calibration + conformal) | `src/uncertainty/` |
| 5 | Adaptation & recovery | `src/adaptation/recalibrate.py` |
| 6 | Write-up | `reports/` |

Each module has working code for the model-agnostic parts (metrics, calibration,
conformal prediction, baselines). The BRFSS-specific parts of `src/data/` are
intentionally left as documented stubs until the target variable and years are
confirmed — filling those in is the first real task once Phase 0 wraps.

## Layout

```
configs/          run configuration (target variable, years, seeds)
src/data/         acquisition + cleaning + temporal split
src/models/       baseline training (Logistic Regression, XGBoost)
src/uncertainty/  calibration, split conformal prediction, failure-detection metrics
src/adaptation/   recalibration on a held-out target-period sample
src/evaluation/   performance/calibration metrics, shift-degradation comparison
data/raw/         downloaded BRFSS files (gitignored)
data/processed/   cleaned/split datasets (gitignored)
reports/          final write-up and figures
docs/             full proposal + any notes from the literature review
```
