# Reliable ML Under Distribution Shift — First Results

**Edmund Eric Gah** · generated from `scripts/run_pipeline.py` and `scripts/make_figures.py` · full numbers in [`phase2-5_results.json`](phase2-5_results.json)

## Summary

This is a first empirical pass at the proposal's central question ([`docs/PROPOSAL.md`](../docs/PROPOSAL.md)): when a health-risk prediction model encounters real distribution shift, does its uncertainty reliably flag which predictions are likely wrong, and can lightweight adaptation recover what's lost? Using four years of real CDC BRFSS survey data (2017 train → 2019 validation → 2021 adaptation sample → 2023 held-out test), the answer so far is: **the shift over this window is real but mild, and the model's uncertainty is genuinely informative regardless.**

## Data

1.68M respondents across 2017/2019/2021/2023 (odd years only — 2018 and 2022 were checked against the real downloaded files and found to completely drop the blood-pressure/cholesterol survey module that year, which would have confounded "distribution shift" with "missing feature"). 15 features covering blood pressure, cholesterol, BMI, smoking, physical activity, general/mental/physical health, and demographics, all cross-year name and coding changes verified against the actual data rather than assumed (see [`src/data/brfss_schema.py`](../src/data/brfss_schema.py)). Diabetes prevalence is stable at 13.6–14.3% across all four years — a basic sanity check that the target recoding is correct.

## Results

### Baseline performance degrades only slightly under 6 years of shift

| Model | AUROC (val 2019) | AUROC (test 2023) | ECE (val) | ECE (test) |
|---|---|---|---|---|
| Logistic Regression | 0.828 | 0.819 | 0.012 | 0.012 |
| Random Forest | 0.832 | 0.824 | 0.013 | 0.012 |
| **XGBoost** | **0.836** | **0.828** | **0.003** | **0.005** |

![ROC curves](figures/roc_curves.png)
![AUROC degradation by model](figures/auroc_degradation.png)

XGBoost is both the strongest performer and the best calibrated out of the box, confirming the proposal's choice of it as the primary model. All three models degrade by roughly the same small amount (~0.01 AUROC) over six years — a real but modest effect, not the dramatic collapse sometimes seen in dataset-shift benchmarks built on synthetic corruption.

### Calibration survives the shift well

![Reliability diagram](figures/reliability_diagram.png)

Raw XGBoost, XGBoost calibrated on the 2019 validation set, and XGBoost recalibrated on a small 2021 sample are nearly indistinguishable on the reliability diagram — all sit close to the diagonal on the 2023 test set. This is itself informative: it means the *raw* model was already reasonably well calibrated even under shift, so calibration's job here is fine-tuning, not damage control.

### Conformal coverage holds close to its target

Split conformal prediction targeted 90% coverage; it achieved exactly 90.0% on its own calibration set (2019, as expected by construction) and **89.7%** on the shifted 2023 test set — a 0.3 percentage-point drop. The theoretical guarantee that conformal coverage can degrade under covariate shift (Tibshirani et al., 2019) is real, but small here, consistent with the mild-shift finding above.

### Uncertainty meaningfully predicts errors, even under shift

![Risk-coverage curve](figures/risk_coverage_curve.png)

This is the most direct answer to the research question. The risk-coverage curve is well-behaved: restricting to the most-confident 20% of 2023 predictions yields under 1% error, while the least-confident predictions carry substantially higher error. Quantitatively, **failure-detection AUROC is 0.815** — well above the 0.5 baseline that would mean uncertainty carries no information about correctness. The model's confidence, even measured under distribution shift, is a genuinely useful signal for deciding which predictions to trust.

### Adaptation improves calibration further, without cost to discrimination

| | ECE on test 2023 |
|---|---|
| Raw XGBoost | 0.0054 |
| Calibrated on val 2019 | 0.0029 |
| Recalibrated on adapt 2021 | **0.0011** |

Recalibrating on a small labeled sample closer to the test period (2021) monotonically improves calibration, and AUROC is essentially unchanged (0.8276 → 0.8275) — exactly the expected behavior, since recalibration reshapes probabilities without changing the ranking of predictions.

### The mild aggregate shift hides a much larger one within a subgroup

![Subgroup AUROC](figures/subgroup_auroc.png)

The headline "shift is mild" finding is an average. Broken out by subgroup on the 2023 test set, sex shows almost no gap (AUROC 0.830 female vs. 0.824 male, equalized-odds difference 0.005), but age band shows a real, substantial one: AUROC falls from **0.836** (18–44) to **0.811** (45–64) to **0.750** (65+) — an equalized-odds difference of **0.138**, roughly 27x the sex gap. The model is meaningfully less discriminative for older adults specifically under the shifted 2023 data, even though the all-ages aggregate looked fine. This is exactly the kind of thing an aggregate-only evaluation would miss, and it's a legitimate fairness finding worth its own discussion in the paper, not just a footnote — full numbers in [`reports/subgroup_fairness.json`](subgroup_fairness.json).

## Discussion

The central hypothesis — that calibrated/conformal uncertainty identifies unreliable predictions under shift, and that lightweight recalibration recovers lost calibration — holds up on this first pass, but the *shift itself* is smaller than the proposal anticipated on average (Section 9, Risk 1). That's a legitimate finding, not a setback: it says something real about how BRFSS-scale national survey data behaves over a 6-year, COVID-spanning window for this particular prediction task. The subgroup analysis above complicates the "mild shift" story usefully, though — it isn't mild for everyone, and the aggregate metric was masking that. It sets up a natural next question: does a *larger* shift (a different geographic split, or comparing states with very different healthcare access) push the conformal-coverage guarantee further, to where it visibly breaks down the way it already has, in effect, for the 65+ subgroup? That would make Risk 3's predicted failure mode observable in the aggregate too, not just within one subgroup.

This also positions the work relative to the closest prior study design: Guo et al. (2022, 2023) split MIMIC-IV hospital records into year groups to study the same kind of temporal shift, but their focus is whether *adaptation* (domain generalization, foundation-model pretraining) recovers performance — not whether the model's own uncertainty can be trusted to flag failure in the first place. That's the gap this project is aimed at, on a different data modality (national population survey, not hospital EHR).

## Limitations

- Single train year (2017) rather than a multi-year training window — chosen to keep the pre/post-shift boundary clean, at the cost of a smaller effective training set (still ~438K rows, not a practical constraint here).
- Missing values handled via median imputation fit on the training year only; more sophisticated imputation was judged not worth the added complexity given missingness rates are modest (<13% for every feature).
- Fruit/vegetable intake and income were dropped from the feature set because their definitions changed across the study window in ways that would confound shift with questionnaire redesign — a reasonable trade-off, but it does mean the feature set is narrower than the full "Diabetes Health Indicators"-style set used elsewhere in the literature.
- This is a methodological study of reliability, not a clinical claim — see the proposal's framing (Section 10).

## In progress

A deep-ensemble / MC-dropout baseline (`scripts/run_deep_ensemble.py`, implementing Lakshminarayanan et al. 2017 and Gal & Ghahramani 2016) is built and correctness-checked on CPU at small scale, but not yet run at the scale intended for the paper — that happens on Exea Labs' AMD Instinct MI300X once access is set up. No deep-ensemble numbers appear in the Results section above for that reason; the smoke-test numbers in `reports/deep_ensemble_results_smoketest.json` are a code check, not a result.

## Reproducing this

```bash
python scripts/download_brfss.py
python scripts/build_dataset.py
python scripts/run_pipeline.py
python scripts/make_figures.py
```
