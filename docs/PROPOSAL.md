# Research Proposal

**Reliable Machine Learning Under Distribution Shift: Uncertainty-Aware Health Risk Prediction**

| | |
|---|---|
| **Researcher** | Edmund Eric Gah |
| **Email** | gahedmund146@gmail.com |
| **Research areas** | Machine Learning · Trustworthy AI · Uncertainty Quantification · Healthcare AI · Distribution Shift |
| **Provided compute** | 4+ CPU cores, 8–16 GB RAM (no GPU required) |

> A formatted, downloadable version of this document is also available at [`docs/proposal.docx`](proposal.docx).

## Executive summary

This proposal investigates whether uncertainty-aware machine learning models can reliably flag their own unreliable predictions when the underlying data distribution shifts — and whether simple adaptation methods can recover lost performance. Using the CDC's Behavioral Risk Factor Surveillance System (BRFSS) and a temporal train/test split, the project trains baseline diabetes-risk classifiers, measures how much predictive performance and calibration degrade across survey years, and tests whether calibrated probabilities and conformal prediction intervals correctly identify which predictions are untrustworthy.

The plan is intentionally scoped to a tractable core (v1) that one researcher can complete with standard tabular ML tools and modest compute, with a clearly separated set of extensions to pursue only if time and mentor guidance allow.

## 1. Background and related work

Prior work shows that standard classifiers become both less accurate and poorly calibrated under dataset shift, and that common uncertainty estimates degrade in reliability precisely when they are needed most (Ovadia et al., 2019). Post-hoc calibration methods such as Platt scaling and isotonic regression are effective in-distribution, but their behavior under shift is comparatively under-studied (Guo et al., 2017; Niculescu-Mizil & Caruana, 2005). Conformal prediction offers distribution-free coverage guarantees under exchangeability (Vovk, Gammerman, & Shafer, 2005), but these guarantees provably degrade under covariate shift, which has motivated weighted and adaptive variants (Tibshirani, Barber, Candès, & Ramdas, 2019). Large empirical benchmarks in this space, such as WILDS (Koh et al., 2021), focus mainly on image and text domains with synthetic or domain-generalization-style shift.

**The gap this project targets:** comparatively little empirical work examines calibration, conformal prediction, and adaptation on tabular, survey-based healthcare data under naturally occurring temporal shift, rather than synthetic corruption. That combination — real-world shift, a classical tabular pipeline, and a healthcare application — is the specific, tractable niche this proposal occupies.

## 2. Research question and hypothesis

**Research question:** When a health-risk prediction model encounters distribution shift, can uncertainty estimation reliably identify predictions that are likely to be incorrect, and can simple adaptation methods restore reliable performance?

**Central hypothesis:** Uncertainty-aware models (calibrated probabilities and conformal prediction) will identify unreliable predictions under distribution shift better than raw model confidence, while lightweight recalibration on a small target-period sample will recover a meaningful portion of lost performance and calibration.

**Sub-questions:**
- How severely does temporal distribution shift degrade predictive performance and calibration on BRFSS diabetes-risk prediction?
- Does high predictive uncertainty (low calibrated confidence, non-coverage in conformal sets) actually correspond to a higher rate of prediction error, especially under shift?
- Can lightweight recalibration using a small labeled sample from the shifted period restore performance and calibration without requiring full retraining?

## 3. Proposed scope: core plan vs. extensions

The original idea covered three shift types, five uncertainty methods, five adaptation methods, and four deliverable types at once — too much for one research engagement to execute well. The table below fixes a tractable core scope (v1) that still directly answers the research question, and separates everything else into clearly labeled extensions to revisit only after the core is working and validated.

| Component | Core (v1) — this proposal | Extension — only if time permits |
|---|---|---|
| Models | XGBoost (primary) + Logistic Regression (interpretable baseline) | Add Random Forest for a 3-way comparison |
| Distribution shift | Temporal shift only — train on earlier BRFSS years, evaluate on later years | Add geographic and demographic/subgroup shift |
| Uncertainty method | Calibrated probabilities (Platt scaling / isotonic regression) + split conformal prediction | Add bootstrap/ensemble uncertainty; weighted or adaptive conformal prediction for shift |
| Adaptation method | Recalibration using a small labeled sample from the shifted (later-year) period | Add threshold adaptation, importance reweighting, fine-tuning/retraining |
| Evaluation | AUROC, AUPRC, Brier score, Expected Calibration Error, conformal coverage, risk–coverage curve | Add Fairlearn subgroup fairness analysis, out-of-distribution detection metrics |
| Deliverable | Reproducible code repository + a short written report | Interactive dashboard; full paper draft for workshop/journal submission |

## 4. Methodology by phase

Total core timeline: roughly 8–11 weeks of focused work. This scales up or down depending on the program's actual duration — to confirm with the mentor at kickoff (see Section 11).

| Phase | Focus | Key activities | Est. | Output |
|---|---|---|---|---|
| 0 | Orientation & literature review | Read core references + 3–5 recent papers on tabular uncertainty; confirm target outcome and BRFSS years with mentor; set up repo and environment | ~1–2 wks | Annotated bibliography, confirmed scope, working environment |
| 1 | Data & preprocessing | Acquire BRFSS years; define diabetes-risk target; clean, encode, and impute; build temporal train/validation/held-out-test split; check leakage and class balance | ~1–2 wks | Reproducible preprocessing pipeline, documented dataset |
| 2 | Baseline models | Train Logistic Regression and XGBoost on training years; evaluate in-distribution on validation years | ~1 wk | Baseline model suite + in-distribution performance report |
| 3 | Distribution shift evaluation | Evaluate frozen baselines on held-out later years; quantify performance and calibration degradation vs. in-distribution | ~1 wk | Shift-degradation analysis with reliability diagrams |
| 4 | Uncertainty quantification | Apply calibration and split conformal prediction; test whether low confidence/non-coverage correlates with actual errors, in-distribution and under shift | ~2 wks | Uncertainty-quality analysis — the core result of the project |
| 5 | Adaptation & recovery | Recalibrate using a small labeled sample from the shifted period; re-evaluate and measure recovery | ~1–2 wks | Before/after adaptation comparison |
| 6 | Write-up & delivery | Finalize repo documentation; write short report; prepare summary for mentor/lab discussion | ~1–2 wks | Final repository + report |

## 5. Dataset

**Primary dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS), a large-scale annual U.S. health survey with demographic, behavioral, and health variables. Its multi-year structure supports a natural temporal-shift design rather than a synthetic or random split.

**Target outcome:** Diabetes-risk prediction (binary), pending confirmation with the mentor — the pipeline generalizes to another BRFSS-derived outcome if preferred.

**Split:** Training on earlier survey years → validation on intermediate years → held-out test on later years, with a small labeled slice of the later-year data reserved specifically for the Phase 5 adaptation step (kept separate from final test evaluation).

**Preprocessing:**
- Data-quality assessment and missing-value analysis
- Removal of irrelevant identifiers; feature selection tied to the research question
- Categorical encoding and standardization where required
- Class-imbalance analysis
- Leakage prevention across the temporal split, documented and version-controlled

## 6. Evaluation metrics (core)

- **Predictive performance:** AUROC, AUPRC, accuracy, precision, recall, F1-score
- **Calibration:** Brier score, Expected Calibration Error (ECE), reliability diagrams
- **Uncertainty quality:** conformal prediction coverage and set size; risk–coverage curves and failure-detection AUROC
- **Robustness under shift:** performance degradation = performance(in-distribution) − performance(shifted)

**Primary success criterion:** calibrated/conformal uncertainty should meaningfully distinguish reliable from unreliable predictions under shift, and recalibration should recover a significant share of lost performance without producing misleading confidence.

## 7. Compute and tooling

The project is deliberately built on classical tabular ML (Logistic Regression, XGBoost) rather than deep learning, so the lab's offered compute is sufficient for every core experiment — no GPU is required.

- **Environment:** Python 3.x, pandas, NumPy, scikit-learn, XGBoost, SciPy, Matplotlib/Seaborn, MAPIE, Jupyter, Git
- **Estimated hardware:** 4+ CPU cores; 8–16 GB RAM; no GPU; ~5–10 GB storage for data and experiment artifacts

Most baseline experiments run in minutes on this hardware. Each experiment run logs random seed, configuration, hyperparameters, dataset version, and results.

## 8. Deliverables

- **Reproducible repository** — data pipeline, model training code, shift and uncertainty experiments, evaluation scripts, and a README with setup and reproduction steps
- **Short written report** — problem framing, method, dataset, results, failure cases, and limitations
- **Extensions (optional)** — interactive dashboard, full paper draft — pursued only after the core is complete and validated

## 9. Risks and mitigations

- **Risk:** the shift between chosen BRFSS years is weaker than expected.
  **Mitigation:** quantify shift magnitude early (Phase 1/3) with distribution-distance statistics before committing further experiments; choose years/outcome with the mentor to maximize signal.
- **Risk:** conformal coverage breaks down under shift.
  **Mitigation:** this is an expected, informative outcome, not a failure — it will be reported directly, with weighted/adaptive conformal prediction flagged as a follow-up extension if it becomes central to the findings.
- **Risk:** the labeled sample available for Phase 5 adaptation is too small or unavailable.
  **Mitigation:** reserve a small labeled slice from within the later-year split itself for recalibration, clearly separated from the final held-out test set and documented as such.

## 10. Expected contribution

This project is framed as a rigorous applied/empirical study rather than a novel-methods contribution: it evaluates existing calibration and conformal prediction techniques on a real, naturally occurring distribution shift in tabular healthcare survey data — a setting that is comparatively underexplored relative to image and text benchmarks. The central question it answers is not "how accurate is the model," but "does the model know when it might be wrong, and can that be restored cheaply when the data changes." For healthcare applications, that distinction matters: an occasionally-wrong model can still be useful, while a confidently-wrong model that cannot recognize unfamiliar situations is much harder to trust. The project will be explicitly presented as a methodological investigation using health-risk prediction as the application domain, not as a claim of clinical validation.

## 11. Open questions for the kickoff meeting

- What is the total time horizon for this research engagement, so the phase estimates in Section 4 can be scaled accordingly?
- Is diabetes-risk prediction the confirmed target outcome, or does the lab prefer a different BRFSS-derived variable?
- Are there specific BRFSS years, regions, or an existing internal preprocessing pipeline the lab wants reused rather than rebuilt?
- What format should the final deliverable take — internal report, workshop paper, thesis chapter?
- Should the project stay strictly within the stated 4+ CPU / 8–16 GB RAM envelope throughout, or is more available if a later stage needs it?
- Is co-authorship or external publication anticipated, and if so, is there a target venue?

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of ICML.*
- Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. *Proceedings of ICML.*
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities With Supervised Learning. *Proceedings of ICML.*
- Ovadia, Y., Fertig, E., Ren, J., et al. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *Advances in NeurIPS.*
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. *Advances in NeurIPS.*
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.

*Full bibliographic details (venue, pages) to be verified against original sources during Phase 0, alongside 3–5 more recent papers to round out the review.*

---

*Prepared by Edmund Eric Gah — Phased Research Proposal for Lab Mentor Review*
