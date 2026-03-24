# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Reviewer 12** -- Expertise: Computational reproducibility, open science standards, REFORMS checklist for ML-based research

**Manuscript**: "Predicting Transboundary Water Conflict Outcomes: An Ordinal Machine Learning Benchmark on the TFDD"

---

## General Assessment

This manuscript presents an ordinal ML benchmark for predicting transboundary water conflict outcomes using the TFDD. The study is methodologically ambitious, integrating eight external data sources, applying ablation analysis with robustness testing, and deploying SHAP temporal decomposition. However, my evaluation focuses exclusively on whether the work can be independently reproduced and whether it meets current best-practice standards for reporting completeness in ML research. I assess the manuscript against the REFORMS checklist (Kapoor & Narayanan, 2023), the ML Reproducibility Checklist, and general open science norms.

The manuscript demonstrates above-average transparency for a domain-science application of ML, particularly in its candid treatment of ablation path-dependence and autoregressive endogeneity. However, several critical reproducibility gaps prevent independent replication, and reporting of computational details falls short of what the REFORMS checklist requires.

---

## Major Concerns

**1. Hyperparameter search spaces and best values are not reported.**

The manuscript states that Optuna Bayesian optimisation was run for 100 trials with basin-grouped 5-fold nested CV, but provides no information about: (a) which hyperparameters were tuned, (b) the search space bounds for each hyperparameter, (c) the best hyperparameter configuration selected, or (d) the objective function used within Optuna (e.g., mean QWK across folds, or a different metric). Extended Data Figure 2 shows a convergence plot, confirming that tuning was performed, but without the search space and selected values, the tuned model cannot be reconstructed. The REFORMS checklist (Item 12: "Report all hyperparameter values, including search ranges and selection criteria") is not satisfied. This is the single largest reproducibility gap in the manuscript.

**2. Computational environment is not specified.**

No information is provided about the Python version, package versions (scikit-learn, XGBoost, LightGBM, Optuna, SHAP), operating system, hardware (CPU/GPU, memory), or random seeds used. The CLAUDE.md file in the repository references a Google Colab origin, but the manuscript does not state whether results were generated on Colab or a local machine. Given that XGBoost and LightGBM can produce different results across library versions and hardware (floating-point non-determinism on different CPU architectures), this omission prevents bitwise reproducibility. At minimum, a requirements.txt, conda environment.yml, or Docker specification should be referenced. REFORMS Item 14 ("Report computational environment") is not met.

**3. Random seed and non-determinism handling are absent.**

There is no mention of random seeds for train/test splitting, cross-validation fold generation, Optuna sampling, or the gradient-boosted models themselves. Without fixed seeds, the 100-trial Optuna search, the bootstrap confidence intervals, and the cross-validation folds are all non-reproducible. The manuscript reports 95% bootstrap CIs from 1,000 resamples but does not state the bootstrap random seed. This means the exact CI bounds reported cannot be verified.

**4. Data preprocessing pipeline is incompletely described.**

While the manuscript describes feature engineering at a conceptual level (zonal aggregation of climate, country-year matching of economic indicators, asymmetry ratios), several critical details are missing:

- How were multi-country events handled for climate zonal statistics? The manuscript mentions "area-weighted means" across basin polygons, but does not specify whether this was computed per-country or per-basin, whether it used the full basin polygon or country-specific basin portions, or what tool was used (rasterstats, xarray, etc.).
- The exact median imputation procedure is not described: was imputation fitted on training data only, or on the full dataset (which would constitute data leakage)?
- The 82-to-45 feature pruning is described procedurally, but the exact list of 82 candidate features is not provided. Extended Data Table 1 lists the 45 retained features but references a supplementary CSV that is not described in sufficient detail.
- The BAR grouping thresholds (negative, zero, 1-3, 4+) are well justified but the code implementing this mapping is not referenced, and the sensitivity analyses for 3-class and 5-class groupings do not specify the exact boundaries used for 5-class until a footnote in Extended Data Table 4.

**5. Temporal and spatial splits cannot be exactly reproduced from the manuscript alone.**

The temporal split is specified (train: pre-1996, validation: 1996-2002, test: 2003-2008), which is good. However, several ambiguities remain: (a) are the boundary years inclusive or exclusive (does a 1996 event go in train or validation)? (b) the basin-grouped 5-fold CV within the training set is described conceptually but the fold assignment method (GroupKFold from sklearn? custom?) is not stated; (c) for continent-level spatial CV (Table 5), the continent assignment for each basin is not described (is it from the TFDD spatial database? manually assigned? what about transcontinental basins like the Tigris-Euphrates?). Without these details, fold assignments cannot be reconstructed.

**6. Variance estimates are incomplete across evaluation metrics.**

Bootstrap 95% CIs are reported for validation QWK in Table 3, but not for: (a) test QWK (the primary reported result of 0.298 has no CI), (b) macro-F1 on either split, (c) per-class metrics in Extended Data Table 2, or (d) spatial CV results in Table 5. The manuscript acknowledges that the bootstrap CIs "capture metric estimation uncertainty but not model training uncertainty" and mentions that "fold-level variance from the nested CV procedure provides a complementary estimate of model instability," but this fold-level variance is never reported. REFORMS Item 17 ("Report uncertainty estimates for all metrics") requires variance estimates on the primary test metric at minimum. The absence of a CI on the headline test QWK of 0.298 is a significant omission.

---

## Minor Concerns

**7. Code repository state is unclear.**

The Data and Code Availability statement links to a GitHub repository, but does not specify: (a) whether the repository contains the analysis code that produced the results in this manuscript (versus the original Colab notebook, which appears to be a different, earlier analysis), (b) which script(s) reproduce which tables and figures, (c) whether a specific commit hash or release tag corresponds to the manuscript results. The REFORMS checklist recommends archiving code in a persistent repository (e.g., Zenodo DOI) rather than linking to a mutable GitHub URL.

**8. Data access barriers are not fully disclosed.**

Several input datasets require registration or have access restrictions that are not mentioned: CRU TS 4.09 requires institutional login or registration; the TFDD spatial database (2024 update) may differ from the version used; World Bank WDI data are accessed "via the wbgapi Python package" but the specific indicator codes and date ranges are not listed. For full reproducibility, the exact API calls or download scripts should be provided or referenced.

**9. The ablation protocol uses LightGBM with "default hyperparameters," but defaults differ across library versions.**

LightGBM defaults have changed across versions (e.g., min_child_samples changed from 20 to 5 in version 4.0). Without specifying the library version, the ablation results in Tables 1 and 2 cannot be replicated.

**10. SHAP version and TreeExplainer configuration are not specified.**

SHAP TreeExplainer behavior differs between versions (e.g., interventional vs. tree-path-dependent feature perturbation). The choice affects SHAP value magnitudes and can change feature rankings. The manuscript does not state which SHAP version or which perturbation method was used.

**11. Bootstrap CI methodology needs fuller specification.**

The manuscript states "percentile bootstrap" with 1,000 resamples, but does not specify whether stratified resampling was used (important given class imbalance), whether the resampling unit was individual predictions or basin-level groups (given within-basin dependence), or whether bias correction (BCa) was applied.

**12. Extended Data Table 1 references external CSV files.**

The manuscript states that the full feature list is "Available in supplementary data file ed_table1_features.csv." This file should be deposited alongside the manuscript or in the code repository. Its current status (whether it exists, where it is hosted) is not clear from the manuscript.

**13. McNemar's test is applied without multiple comparison correction.**

Extended Data Figure 3 reports raw p-values for pairwise model comparisons. With 6 models and 15 pairwise comparisons, a Bonferroni or FDR correction should be applied, or the absence of correction should be explicitly justified.

**14. The ordinal regression baselines lack test-set evaluation.**

Table 3 shows test QWK only for XGBoost models. LogisticAT and OrdinalRidge are evaluated on the validation set only. Without test-set metrics for all models, the claim that XGBoost "outperforms ordinal regression baselines" cannot be verified on the held-out data.

---

## REFORMS Checklist Summary

| REFORMS Item | Status |
|:---|:---|
| Problem formulation and context | Met |
| Data description | Partially met (missing indicator codes, download details) |
| Data preprocessing | Partially met (imputation leakage unclear, pipeline gaps) |
| Feature engineering | Mostly met (good conceptual description, missing implementation detail) |
| Train/validation/test split | Mostly met (boundary ambiguity, fold assignment unspecified) |
| Model selection and justification | Met |
| Hyperparameter tuning | NOT met (search spaces and best values absent) |
| Evaluation metrics and uncertainty | Partially met (no CI on test metrics, no fold-level variance) |
| Baseline comparisons | Partially met (baselines lack test evaluation) |
| Code and data availability | Partially met (no persistent archive, no version pinning) |
| Computational environment | NOT met |
| Reproducibility artifacts (seeds, versions) | NOT met |

---

## Recommendations

Before acceptance, the authors should:

1. Report the full Optuna search space (parameter names, bounds, distributions) and the best hyperparameter configuration for all tuned models, either in the Methods section or as an Extended Data table.
2. Add a computational environment specification: Python version, key package versions (XGBoost, LightGBM, SHAP, Optuna, scikit-learn, geopandas), and all random seeds.
3. Provide bootstrap or permutation-based 95% CIs for the test QWK and test macro-F1 of the primary model, and report fold-level standard deviations from the nested CV.
4. Clarify the imputation procedure (fit on training data only, or full dataset) and specify the exact feature list for the 82-candidate pool.
5. Archive the code at a persistent DOI (Zenodo) with a tagged release corresponding to the manuscript, and include a script-to-figure mapping.
6. Evaluate ordinal regression baselines on the test set for a complete model comparison.

These revisions would bring the manuscript into compliance with REFORMS standards and enable independent replication of the reported results.

---

*Reviewed: 2026-03-24*
