# Reliable Machine Learning Under Distribution Shift: Uncertainty-Aware Diabetes Risk Prediction from National Health Survey Data

**Edmund Eric Gah**
*Draft — all sections, including 4.5, now reflect completed experiments on real data.*

## Abstract

Machine-learning models evaluated only on held-out data from their own training distribution can look reliable and then fail silently once deployed, as patient populations and data-collection practices drift across time, geography, and demographic groups. This paper investigates whether calibrated probabilities and conformal prediction reliably flag which predictions of a health-risk model are untrustworthy once its input distribution shifts, using diabetes-risk prediction on the CDC's Behavioral Risk Factor Surveillance System (BRFSS) as the application. Using three real, non-synthetic shift settings — a 6-year temporal shift (2017→2023), demographic subgroups within it, and a single-year geographic shift across US Census regions — together with a fourth analysis testing whether a known remedy corrects the coverage loss found in the third, we find that (1) aggregate temporal degradation is mild, but this average conceals a substantial fairness gap by age (AUROC 0.836 at ages 18–44 versus 0.750 at 65+); (2) geographic shift, isolated from time, degrades calibration and conformal coverage far more than six years of temporal shift did (conformal coverage 90.0%→87.3% for the South versus 90.0%→89.7% temporally); (3) the model's own uncertainty remains informative under every shift tested (failure-detection AUROC 0.80–0.82 throughout); and (4) weighted conformal prediction (Tibshirani et al., 2019) recovers roughly half the coverage lost to geographic shift. Together these results argue that shift *magnitude*, not merely shift's presence, determines whether uncertainty-aware safeguards hold — and that evaluating only one shift dimension, or only in aggregate, can miss the failure modes that matter most.

## 1. Introduction

A model that occasionally makes an incorrect prediction can still be useful in a clinical or public-health setting; a model that is confidently wrong, and unable to recognize that it has left the population it was trained on, is considerably more dangerous. Standard machine-learning evaluation — a single train/test split, an aggregate accuracy number — does not distinguish these two cases. The gap between them is especially consequential in healthcare, where the populations, practices, and even survey instruments a model encounters after deployment routinely differ from those it was trained on.

This paper asks a more specific question than "how accurate is the model": when a health-risk prediction model encounters a distribution shift, does its own uncertainty reliably identify which of its predictions are likely wrong, and can that reliability be restored cheaply once it degrades? We study this using diabetes-risk prediction on BRFSS, a large annual US health survey whose multi-year, multi-region structure supports studying *real* distribution shift directly, rather than the synthetic corruptions common in the uncertainty-quantification literature (Ovadia et al., 2019; Koh et al., 2021).

Our contribution is empirical rather than methodological: we apply established techniques — post-hoc calibration, split and weighted conformal prediction, standard fairness metrics — rigorously to a real, underexplored setting, rather than proposing new ones. Concretely:

1. A 6-year temporal shift analysis (2017 train → 2019/2021/2023 evaluation) shows performance and calibration degrade only mildly on average.
2. A subgroup fairness analysis shows that this average conceals a large age-related gap the aggregate evaluation misses entirely.
3. A geographic shift analysis, isolated from time within a single year, shows geography to be a substantially larger shift than six years of time — large enough to measurably strain the conformal coverage guarantee.
4. A weighted conformal prediction analysis shows that the theoretically-prescribed fix for that strain works, partially.

## 2. Related Work

Prior work shows that standard classifiers become both less accurate and poorly calibrated under dataset shift, and that common uncertainty estimates — including deep ensembles and MC-dropout — degrade in reliability precisely when they are needed most (Ovadia et al., 2019; Lakshminarayanan et al., 2017; Gal & Ghahramani, 2016). Post-hoc calibration methods such as Platt scaling and isotonic regression are effective in-distribution, but their behavior under shift is comparatively under-studied (Guo et al., 2017; Niculescu-Mizil & Caruana, 2005). Conformal prediction offers distribution-free coverage guarantees under exchangeability (Vovk, Gammerman, & Shafer, 2005), but these guarantees provably degrade under covariate shift, which has motivated weighted and adaptive variants (Tibshirani, Barber, Candès, & Ramdas, 2019) — the method we apply directly in Section 4.4. Large empirical benchmarks in this space, such as WILDS (Koh et al., 2021), focus mainly on image and text domains with synthetic or domain-generalization-style shift. Uncertainty for gradient-boosted trees specifically — our primary model family — has its own, separate literature (Malinin, Prokhorenkova, & Ustimenko, 2021).

Closest to our design are two studies that split real electronic health record (EHR) data (MIMIC-IV) into year groups to study temporal shift directly: Guo et al. (2022) benchmarks domain generalization and adaptation algorithms across four year groups on mortality, length-of-stay, sepsis, and ventilation prediction, and Guo et al. (2023) shows that EHR foundation models pretrained across year groups improve robustness to the same shift. Both center on *adaptation* — recovering performance after shift — rather than on whether the model's own uncertainty can be trusted to flag failure in the first place, which is our focus here, and both use hospital EHR data rather than a national population survey. That combination — real-world shift in survey data, a classical tabular pipeline, and an explicit focus on uncertainty-as-failure-detector across *multiple* shift dimensions rather than one — is where this work sits relative to prior literature.

## 3. Data and Methods

### 3.1 Dataset

The Behavioral Risk Factor Surveillance System (BRFSS) is a large annual US health survey conducted by the CDC. We use four survey years — 2017, 2019, 2021, and 2023 — restricted to odd years for a reason discovered only by inspecting the raw downloaded files rather than the codebook alone: 2018 and 2022 completely omit BRFSS's blood-pressure/cholesterol module that year, a biennial rotation, and using either as an evaluation year would have confounded "distribution shift" with "missing feature." The target is a binary diabetes indicator (`DIABETE3` in 2017, renamed `DIABETE4` from 2019 onward, with confirmed identical response coding across the rename). Fifteen features cover blood pressure, cholesterol, BMI, smoking, physical activity, self-rated general health, mental and physical health days, and basic demographics; every cross-year name and coding change was verified against the real data rather than assumed (full schema and rationale in `src/data/brfss_schema.py`). Fruit/vegetable intake and income were excluded because their definitions changed within the study window in ways that would themselves have constituted a confound. The combined dataset totals 1,684,646 respondents, and diabetes prevalence is stable at 13.6–14.3% across all four years — a basic sanity check confirming the target was constructed correctly.

### 3.2 Models

Three baselines are trained: Logistic Regression as an interpretable linear reference, Random Forest, and XGBoost as the primary model — chosen for its established strength on tabular data, and confirmed empirically in Section 4.1 to be both the most accurate and the best-calibrated of the three. A ten-member deep ensemble and an MC-dropout network (Lakshminarayanan et al., 2017; Gal & Ghahramani, 2016), each trained for 50 epochs, provide a fourth comparison point (Section 4.5), trained on an AMD Instinct MI300X.

### 3.3 Uncertainty Quantification

Post-hoc calibration uses isotonic regression fit on a held-out split. Conformal prediction uses the split-conformal method with a least-ambiguous-set nonconformity score, targeting 90% coverage. Weighted conformal prediction (Section 4.4) reweights calibration nonconformity scores by an estimated covariate density ratio between the calibration and target domains, estimated via a logistic-regression domain classifier, following Tibshirani et al. (2019); with calibration sets in the tens of thousands, we use the standard large-sample simplification of dropping each test point's own (negligible) contribution to the normalizing constant, so a single weighted threshold is computed rather than one per test point.

### 3.4 Evaluation Metrics

Discrimination: AUROC, AUPRC. Calibration: Brier score, Expected Calibration Error (ECE). Uncertainty quality: conformal coverage and mean prediction-set size, and failure-detection AUROC — whether a model's uncertainty score (here, distance from a 0.5 decision boundary) discriminates between its correct and incorrect predictions. Fairness: Fairlearn's demographic parity and equalized odds differences.

### 3.5 Experimental Designs

**Temporal shift** (Section 4.1): train on 2017, calibrate/validate on 2019, hold out a small recalibration sample from 2021, evaluate final performance on 2023 — isolating a 6-year, COVID-spanning shift while keeping geography constant.

**Subgroup fairness** (Section 4.2): the 2023 test set broken out by sex and a three-band age grouping (18–44, 45–64, 65+), checking whether the aggregate temporal result conceals a subgroup-specific one.

**Geographic shift** (Section 4.3): to isolate geography from time, this analysis uses a *single* year (2023) throughout. XGBoost is trained on Northeast respondents only (US Census region, mapped from BRFSS's state FIPS code against the official Census Bureau reference table), then evaluated on a held-out Northeast slice (in-region reference) versus the Midwest, South, and West (out-of-region).

**Weighted conformal prediction** (Section 4.4): applied to the geographic-shift setup above, comparing unweighted and weighted conformal coverage for each out-of-region evaluation.

## 4. Results

### 4.1 Temporal Distribution Shift

| Model | AUROC (val 2019) | AUROC (test 2023) | ECE (val) | ECE (test) |
|---|---|---|---|---|
| Logistic Regression | 0.828 | 0.819 | 0.012 | 0.012 |
| Random Forest | 0.832 | 0.824 | 0.013 | 0.012 |
| **XGBoost** | **0.836** | **0.828** | **0.003** | **0.005** |

![ROC curves](../reports/figures/roc_curves.png)

XGBoost is both the strongest performer and the best calibrated out of the box. All three models degrade by roughly the same small amount (~0.01 AUROC) over six years. Split conformal prediction targeted 90% coverage and achieved 89.7% on the shifted 2023 test set — a 0.3-point drop. Failure-detection AUROC is 0.815: restricting to the most-confident 20% of 2023 predictions yields under 1% error (risk-coverage curve below), meaning the model's confidence remains informative even under shift.

![Risk-coverage curve](../reports/figures/risk_coverage_curve.png)

![Reliability diagram](../reports/figures/reliability_diagram.png)

Recalibrating on a small 2021 sample tightens calibration further (ECE 0.0054 → 0.0011) with no cost to discrimination (AUROC 0.8276 → 0.8275) — the expected behavior, since recalibration reshapes probabilities without changing prediction ranking.

### 4.2 Subgroup Fairness Under Shift

![Subgroup AUROC](../reports/figures/subgroup_auroc.png)

Broken out by subgroup on the 2023 test set, sex shows almost no gap (AUROC 0.830 female vs. 0.824 male; equalized-odds difference 0.005), but age band shows a substantial one: AUROC falls from 0.836 (18–44) to 0.811 (45–64) to **0.750** (65+) — an equalized-odds difference of 0.138, roughly 27x the sex gap. The aggregate temporal result in Section 4.1 does not surface this at all.

### 4.3 Geographic Distribution Shift

| Region | AUROC | ECE (calibrated on Northeast) | Conformal coverage (target 90%) |
|---|---|---|---|
| Northeast (in-region) | 0.827 | 0.000 | 90.0% |
| Midwest | 0.819 | 0.007 | 88.6% |
| South | 0.816 | **0.018** | **87.3%** |
| West | 0.827 | 0.004 | 90.2% |

![Geographic shift](../reports/figures/geographic_shift.png)

Raw discrimination barely moves, echoing Section 4.1. Calibration and conformal coverage tell a different story: the South's conformal coverage drop (90.0%→87.3%, 2.7 points) is roughly 9x the drop under the full 6-year temporal shift, and its calibration error is nearly 18x worse than in-region. This aligns with an independently documented pattern — the Southern US has substantially higher diabetes prevalence than the Northeast (16.9% vs. ~12.3% in this data, consistent with the CDC's "diabetes belt") — so a model calibrated on the Northeast's lower base rate is systematically overconfident when applied to the South's higher one. Failure-detection AUROC still holds up reasonably everywhere (0.80–0.82).

### 4.4 Weighted Conformal Prediction

![Weighted conformal](../reports/figures/weighted_conformal.png)

| Region | Unweighted coverage | Weighted coverage | Mean set size (unweighted → weighted) |
|---|---|---|---|
| Midwest | 88.6% | **89.6%** | 1.06 → 1.09 |
| South | 87.3% | **88.7%** | 1.07 → 1.11 |
| West | 90.2% | 90.0% | 1.05 → 1.04 |

Weighting closes roughly half the South's coverage gap and nearly all of the Midwest's, at the cost of modestly larger prediction sets — the expected trade-off. For the West, where there was barely any gap, weighting has essentially no effect, a useful sanity check that the method does not manufacture a problem where none exists. It does not fully close the South's gap: the density-ratio estimate is only as good as the domain classifier estimating it, and this particular covariate shift is large.

### 4.5 Deep Ensembles and MC-Dropout

| Model | AUROC (test 2023) | ECE, raw (test 2023) | Failure-detection AUROC |
|---|---|---|---|
| Deep ensemble (10 members) | 0.827 | 0.0099 | 0.800 (member disagreement) |
| MC-dropout (single net) | — | — | 0.786 |
| XGBoost (raw / calibrated) | 0.828 | 0.0054 | 0.815 (calibrated) |

![Deep ensemble comparison](../reports/figures/deep_ensemble_comparison.png)

Trained on an AMD Instinct MI300X (10 members, 50 epochs each), the deep ensemble matches XGBoost's discrimination almost exactly (AUROC 0.827 vs. 0.828) despite using none of gradient boosting's tree-structure inductive bias — a small MLP ensemble is enough to reach the same ceiling on this feature set. Its raw calibration is worse than XGBoost's raw calibration (ECE 0.0099 vs. 0.0054), consistent with Ovadia et al. (2019)'s finding that ensembles are reasonably but not perfectly calibrated out of the box. Ensemble disagreement — the variance across members' predictions — is a genuinely useful uncertainty signal (failure-detection AUROC 0.800), and outperforms MC-dropout on the same trained network (0.786), again consistent with the literature's general finding that ensembles give better-behaved uncertainty than dropout-based approximation alone. Both remain informative, in the same range as XGBoost's calibrated signal (0.815) from Section 4.1 — the finding that uncertainty survives this shift is not specific to gradient-boosted trees.

## 5. Discussion

The central pattern across Sections 4.1–4.4 is that **shift magnitude, not merely shift's presence, determines how much of the reliability story holds** — and that this magnitude varies enormously depending on which dimension of shift is measured and at what level of aggregation. The temporal shift, averaged over the whole population, looked mild. It wasn't mild for the 65+ subgroup, and it wasn't mild when geography rather than time was the shift axis: the South's conformal coverage came measurably closer to the kind of breakdown the literature predicts (Tibshirani et al., 2019) than six years of temporal drift ever did. An evaluation that stopped at the aggregate temporal result would have concluded the model was essentially shift-robust. It isn't — not for the 65+ subgroup, and not for the South.

This has a direct methodological implication for how uncertainty-aware systems should be evaluated before deployment: a single shift axis, measured only in aggregate, is not sufficient evidence of reliability. The subgroup and geographic analyses here were comparatively cheap to run once the core pipeline existed — the marginal cost of checking multiple shift dimensions is small relative to the risk of missing the one that matters.

The weighted conformal result (Section 4.4) is encouraging but incomplete in a specific, informative way: it demonstrates that the theoretically-correct response to detected covariate shift genuinely helps, while also demonstrating its limit — a simple domain classifier cannot fully correct for a shift this large from so few calibration examples. That gap is itself worth reporting, rather than papering over with a stronger domain classifier tuned to close it; a more expressive density-ratio estimator is a natural next step, but the honest finding at this stage is partial, not complete, recovery.

The deep ensemble comparison (Section 4.5) adds one more piece to this picture: uncertainty surviving distribution shift is not an artifact specific to gradient-boosted trees. A neural network ensemble with no tree-structure inductive bias reaches the same discrimination ceiling as XGBoost and produces a comparably informative uncertainty signal via member disagreement, even though its raw calibration is worse out of the box. That the central finding replicates across two structurally different model families is modest additional evidence that it reflects something about the *data and shift*, rather than an idiosyncrasy of one algorithm.

## 6. Limitations

- Single train year (2017) rather than a multi-year training window, chosen to keep the pre/post-shift boundary clean.
- Missing values handled via median imputation fit on the training split only; missingness rates are modest (<13% for every feature).
- The feature set omits fruit/vegetable intake and income due to within-window definitional changes, making it narrower than the "Diabetes Health Indicators"-style feature sets used elsewhere in the literature.
- The geographic-shift domain classifier (Section 3.3/4.4) is a plain logistic regression on the same 15 features; a more expressive estimator might close more of the coverage gap.
- This is a methodological study of reliability under shift, not a clinical validation — health-risk prediction is the application domain, not a deployment claim.

## 7. Conclusion

Uncertainty-aware evaluation of a real diabetes-risk model under real distribution shift shows that calibration and conformal coverage are far more sensitive to shift magnitude than raw discrimination is, that this sensitivity is easy to miss when only one shift dimension or only the aggregate is examined, and that a theoretically-motivated fix for detected coverage loss (weighted conformal prediction) works, partially. The practical recommendation this supports is straightforward: evaluate uncertainty-aware systems across multiple, independent shift dimensions — not just time — before treating aggregate stability as evidence of reliability.

## Acknowledgments

This research is conducted with mentorship from Avneh Singh Bhatia (Exea Labs). The deep-ensemble and MC-dropout experiments in Section 4.5 ran on an AMD Instinct MI300X accelerator, provided by AMD via Exea Labs.

## References

*All entries verified against primary sources (arXiv/DOI/publisher), not from memory alone.*

- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *Proceedings of ICML*, PMLR 48:1050–1059.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of ICML*, PMLR 70:1321–1330. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Guo, L. L., Pfohl, S. R., Fries, J., Johnson, A., Posada, J., Aftandilian, C., Shah, N., & Sung, L. (2022). Evaluation of Domain Generalization and Adaptation on Improving Model Robustness to Temporal Dataset Shift in Clinical Medicine. *Scientific Reports*, 12, 2726.
- Guo, L. L., Steinberg, E., Fleming, S. L., Posada, J., Lemmon, J., Pfohl, S. R., Shah, N., Fries, J., & Sung, L. (2023). EHR Foundation Models Improve Robustness in the Presence of Temporal Distribution Shift. *Scientific Reports*, 13, 3767.
- Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., et al. (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. *Proceedings of ICML.* [arXiv:2012.07421](https://arxiv.org/abs/2012.07421)
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles. *Advances in NeurIPS 30*, 6402–6413.
- Malinin, A., Prokhorenkova, L., & Ustimenko, A. (2021). Uncertainty in Gradient Boosting via Ensembles. *ICLR 2021.* [arXiv:2006.10562](https://arxiv.org/abs/2006.10562)
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities With Supervised Learning. *Proceedings of the 22nd ICML.* [DOI:10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)
- Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *Advances in NeurIPS 32.* [arXiv:1906.02530](https://arxiv.org/abs/1906.02530)
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. *Advances in NeurIPS 32*, 2526–2536. [arXiv:1904.06019](https://arxiv.org/abs/1904.06019)
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer. ISBN 978-0-387-00152-4.
