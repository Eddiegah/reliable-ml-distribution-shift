# Reliable ML Under Distribution Shift
### Uncertainty-Aware Health Risk Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active--research-orange)

**Does a model know when it might be wrong?**

Machine-learning models can look excellent in development and still fail quietly
in deployment — not by crashing, but by staying confidently wrong once the data
they see no longer looks like the data they were trained on. In healthcare,
that gap is dangerous: patient populations, practices, and data collection
drift across time, geography, and demographics, and a model that can't tell
you it's out of its depth is a model you can't trust.

This project trains diabetes-risk classifiers on the CDC's [Behavioral Risk
Factor Surveillance System](https://www.cdc.gov/brfss/) (BRFSS), evaluates them
under **real temporal distribution shift** (train on earlier survey years,
test on later ones — not synthetic noise), and asks a sharper question than
standard evaluation does: when the model's confidence is high, is it actually
*earned*? Calibrated probabilities and conformal prediction are tested as the
mechanism for flagging unreliable predictions, and lightweight recalibration
is tested as the fix once shift is detected.

📄 Full proposal: [`docs/proposal.docx`](docs/proposal.docx)

---

## Research question

> When a health-risk prediction model encounters distribution shift, can
> uncertainty estimation reliably identify predictions that are likely to be
> incorrect — and can adaptation methods restore reliable performance?

## Status — open items before Phase 1

`configs/default.yaml` has a few fields marked `TBD`. These are confirmed with
the lab before the pipeline runs on real data (see Section 11 of the proposal):

- [ ] Target variable (which BRFSS diabetes field, and which years it's coded consistently)
- [ ] Train / validation / test / adaptation-sample years
- [ ] Total time horizon for the engagement (scales the phase estimates below)

## Roadmap

| Phase | Focus | Entry point |
|---|---|---|
| 0 | Literature review | `docs/` |
| 1 | Data acquisition & preprocessing | `src/data/acquire.py`, `src/data/preprocess.py` |
| 2 | Baseline models (Logistic Regression, XGBoost) | `src/models/baselines.py` |
| 3 | Distribution shift evaluation | `src/evaluation/metrics.py` (`performance_degradation`) |
| 4 | Uncertainty quantification (calibration + conformal) | `src/uncertainty/` |
| 5 | Adaptation & recovery | `src/adaptation/recalibrate.py` |
| 6 | Write-up | `reports/` |

The model-agnostic parts (metrics, calibration, conformal prediction, baseline
training) are implemented and ready to run. The BRFSS-specific parts of
`src/data/` are intentionally left as documented stubs until the target
variable and years are confirmed with the lab — filling those in is the first
real task once Phase 0 wraps.

## Setup

Requires Python 3.10+ (type hints use the `X | Y` syntax).

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Repo layout

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
docs/             full proposal + literature review notes
```

## Contributors

| | |
|---|---|
| **Edmund Eric Gah** | Researcher & maintainer — [gahedmund146@gmail.com](mailto:gahedmund146@gmail.com) |

Contributions and mentor guidance welcome once the project is underway — open
an issue or reach out directly.

## License

Released under the [MIT License](LICENSE).
