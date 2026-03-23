# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Journal**: *Global Environmental Change*

**Reviewer**: Reviewer 4

**Expertise**: Machine learning generalization, temporal prediction, applied ML for social science

**Date**: 23 March 2026

---

## 1. Summary

This paper constructs the first ordinal-aware machine learning benchmark for predicting the intensity of transboundary water conflict events on the BAR scale, using the TFDD (6,805 events, 1948-2008) enriched with eight external data sources. Through a sequential ablation protocol, the authors find that economic indicators and temporal dynamics are candidate predictors of event-level conflict intensity, while basin-averaged annual climate variables do not improve prediction. An Optuna-tuned XGBoost classifier achieves QWK of 0.502 on validation data (1996-2002) but degrades to 0.290 on a held-out test set (2003-2008), a 42% decline that the authors transparently diagnose. SHAP temporal decomposition reveals increasing importance of treaty formation rate in the post-2000 era relative to the Cold War period.

## 2. Overall Assessment

**Recommendation: Major Revision**

This manuscript makes a genuine methodological contribution to the hydropolitics literature. The ordinal treatment of the BAR scale, the multi-source feature integration, the ablation protocol with robustness testing, and the unusually candid self-assessment of the validation-to-test gap all reflect careful, mature scientific thinking. The paper is also commendably well situated in the hydropolitics literature, drawing on a breadth of theory (hydro-hegemony, TWINS, institutional resilience) that many applied ML papers neglect entirely.

However, I have substantive concerns about model generalization, the interpretation of autoregressive features, the ablation methodology, and whether several conclusions are appropriately calibrated to the strength of the evidence. The 42% validation-to-test degradation is not merely a limitation to be acknowledged; it is the central empirical result, and its implications for every other claim in the paper need to be more thoroughly confronted. Below I detail major and minor concerns, followed by specific questions.

## 3. Major Concerns

**1. The 42% validation-to-test gap undermines the evidentiary basis for most substantive claims.**

The authors deserve credit for reporting this gap transparently and investigating multiple explanations (distributional shift, Optuna overfitting, regime change). However, the diagnostic analysis, while welcome, does not go far enough. Consider the implications: a QWK of 0.290 on the test set means the model captures only modest ordinal structure beyond chance. Yet the SHAP analysis, the ablation conclusions, the discussion of climate irrelevance, and the institutional-change hypothesis are all derived from a model whose learned representations do not transfer to the test period. The authors perform SHAP on the validation-optimised model, but if that model's patterns are substantially epoch-specific, the SHAP importance rankings may characterise the 1996-2002 validation era rather than stable, generalisable drivers of conflict dynamics.

I request the following:

(a) Report SHAP analysis separately on the test set predictions and compare the feature importance rankings to the validation-set SHAP. If the rankings are materially different, the discussion of "drivers" must be hedged accordingly.

(b) The rolling-window analysis (15-year train, 5-year predict) is mentioned briefly. Report the full table of QWK values per window, and compute SHAP rankings for each window. If treaty formation rate is not consistently important across windows, the 56.5% increase claim needs substantial qualification.

(c) Report the calibration of predicted class probabilities on both validation and test sets. A model that achieves QWK of 0.502 through well-calibrated ordinal ranking will degrade differently than one that achieves it through overconfident majority-class prediction. Reliability diagrams or expected calibration error would clarify the failure mode.

**2. Autoregressive feature endogeneity is more severe than acknowledged, and the ablation analysis of these features is insufficient.**

The authors correctly identify the endogeneity risk in cooperation momentum, events in the prior 5 years, and event escalation, and Table 4 shows that removing these features actually improves test performance. This is an important result that deserves more prominent treatment. However, the analysis has a critical gap: the autoregressive features are not merely endogenous in a statistical sense; they create a fundamental information leakage pathway that undermines temporal prediction validity.

Cooperation momentum is the rolling mean BAR of prior events in the same basin. In a temporal prediction setting, this feature encodes the label distribution of the recent past. If the BAR distribution shifts between training and test periods (which the authors confirm it does), then the cooperation momentum feature carries a distributional signature that is informative during validation (where the shift is smaller) but misleading during testing (where the shift is larger). This is not standard endogeneity; it is a form of target leakage through a temporally lagged proxy that degrades precisely when the model most needs to generalise.

The test in Table 4 uses XGBoost with default hyperparameters, not the Optuna-tuned model. Given that the Optuna-tuned model is the paper's headline result (QWK 0.502), the authors must report the autoregressive ablation on the tuned model as well. It is entirely possible that Optuna's 100-trial search exploits the autoregressive features more aggressively than default hyperparameters, in which case the tuned model's validation performance may be even more inflated by endogenous information.

Furthermore, the paper's best-performing ablation ordering (shuffled: baseline + economic + temporal, QWK 0.412) includes the temporal group. The "final" 45-feature set used for model comparison in Table 3 also includes temporal features. This means the headline model comparison is conducted on a feature set that includes endogenous features. I strongly recommend that Table 3 be re-run on the feature set without autoregressive features and that the resulting test QWK be reported as the primary result.

**3. The path-dependent ablation protocol does not support the strength of the feature group rankings.**

I appreciate the robustness analysis across three orderings (Table 2). However, three orderings out of 7! = 5,040 possible orderings is a very sparse sample of the permutation space. The three orderings tested produced three different retained sets (baseline + AQUASTAT + asymmetry; baseline + temporal + economic + climate; baseline + economic + temporal), with validation QWK values ranging from 0.400 to 0.412. This range is narrow, but the retained feature groups are qualitatively different, which means the ablation cannot reliably distinguish which groups matter.

The fundamental problem is that forward sequential ablation conflates marginal and conditional importance. A feature group's delta depends on what is already in the model, and correlated groups will compete for the same variance. Climate and economic features may explain overlapping variance in conflict outcomes, so whichever enters first "wins" and the second is discarded.

I request:

(a) Report the all-subsets analysis for the seven feature groups. With only seven groups, there are 2^7 = 128 possible subsets, which is computationally trivial for a LightGBM model with default hyperparameters. This would provide an exhaustive map of feature group contributions and their interactions.

(b) Alternatively, if all-subsets is impractical for some reason, at minimum report a Shapley-value decomposition at the feature group level (not individual features). This would provide a theoretically grounded, order-independent estimate of each group's contribution to validation QWK.

(c) Report the test QWK for all three ablation orderings' retained sets. If the ordering that retains climate (reversed: QWK 0.400 on validation) achieves comparable or better test performance than the ordering that discards climate (shuffled: 0.412 on validation), then the claim that climate variables "do not improve prediction" is further weakened.

**4. The model comparison is confounded by feature set selection.**

Table 3 reports all models on the "45-feature set (baseline + economic + temporal, from the best-performing ablation ordering)." However, the ablation was conducted using LightGBM with default hyperparameters. The feature set that is optimal for LightGBM-default is not necessarily optimal for XGBoost-tuned, OrdinalRidge, or LogisticAT. Linear ordinal models may benefit from different feature subsets than tree-based models, particularly if multicollinearity is present (which is likely given the asymmetry ratios and economic indicators). Running all models on a feature set optimised for one model family introduces a systematic bias in favour of gradient-boosted trees.

At minimum, the authors should report the ordinal regression models on the full 82-feature set as well as the pruned 45-feature set, and discuss whether the pruning systematically disadvantaged the linear models.

**5. The temporal SHAP decomposition is descriptive, not inferential, and the 56.5% figure requires uncertainty quantification.**

The claim that treaty formation rate "increased 56.5% in importance from the Cold War to the post-2000 era" is presented as a key finding supporting the institutional-change hypothesis. However, SHAP values are point estimates from a single trained model applied to subsets of data from different eras. The apparent shift in importance could arise from: (a) genuine shifts in the data-generating process, (b) differences in feature distributions across eras (treaty formation rate may simply have higher variance post-2000, mechanically increasing SHAP magnitude), (c) sample size differences across eras affecting SHAP stability, or (d) model misspecification that manifests differently across eras.

Without uncertainty quantification on the era-specific SHAP values, the 56.5% figure is uninterpretable. I request bootstrap confidence intervals on the era-specific mean |SHAP| values, computed by resampling events within each era. If the confidence intervals for the Cold War and post-2000 treaty formation rate importance overlap, the "shift" is not statistically supported. Additionally, the authors should report the number of events in each era and verify that the post-2000 era, which covers only 2000-2008 and likely has fewer events, is not driving inflated SHAP estimates through small-sample instability.

## 4. Minor Concerns

1. **QWK as the sole primary metric for ordinal evaluation.** QWK penalises large ordinal errors but is insensitive to the direction of misclassification. For policy applications, predicting "strong cooperation" when the true outcome is "conflict" (a 3-class error) is much more consequential than predicting "mild cooperation" when the true outcome is "strong cooperation." An asymmetric cost-weighted metric, or at minimum a cost-sensitive confusion matrix, should be reported.

2. **The neutral class (4.0%, 34 test samples) distorts macro-F1 and per-class metrics.** With only 34 test samples, any per-class metric for the neutral class is essentially noise. The authors should consider reporting metrics both with and without the neutral class, or collapsing it into one of the adjacent classes and conducting a sensitivity analysis on this choice.

3. **Imputation of governance features.** The authors note 60-64% missingness in governance indicators and use median imputation. The brief mention that native NaN handling and missingness indicators improved test QWK by +0.020 and +0.022 is buried in the limitations section. Given that these improvements are comparable in magnitude to the ablation deltas used to retain or discard entire feature groups (the retention threshold is 0.005), the imputation strategy is not a minor methodological detail but a decision that could change the ablation outcome for governance. This analysis should be moved to the main results and the governance ablation re-run with proper NaN handling.

4. **McNemar's test for pairwise model comparison** is referenced (Extended Data Figure 3) but not reported in the main text. For a methods-focused paper, at least the key pairwise comparison (XGBoost-tuned vs. next-best model) should include the test statistic and p-value in the main results.

5. **The continent-level spatial CV result** (mean QWK 0.248, range 0.062-0.417) is mentioned only in the limitations. This range is enormous and suggests that the model performs very differently across geographic contexts. The continents with QWK = 0.062 deserve individual discussion: which continents are they, and why does the model fail there?

6. **Raw year as a SHAP feature.** The authors acknowledge that year of occurrence encodes secular trends as a proxy for unmodelled confounders. However, including raw year in a temporal prediction model is problematic beyond interpretability: it means the model cannot extrapolate to years outside the training range. For the test set (2003-2008), years are within the training range (pre-1996) only if we consider the full dataset, but the model was trained on pre-1996 data only. How does XGBoost handle test-set year values (2003-2008) that are outside the training range (pre-1996)? Tree-based models will assign all post-1996 years to the same leaf as the maximum training-set year, effectively treating all post-1996 events identically with respect to this feature. This is a concrete mechanism that could contribute to the validation-to-test gap and should be explicitly discussed.

7. **The sensitivity analysis with 3-class and 5-class groupings** is mentioned but relegated to Extended Data. Given that the 4-class grouping is a discretisation choice that may affect all downstream results, the key finding (5-class marginally improves validation QWK but no grouping improves test performance) should be in the main text.

8. **Figure references.** Several figures (Fig. 1b, 1c, 2a, 2b, 3, 4) are referenced but I cannot evaluate them from the manuscript text alone. Ensure that confusion matrices for both validation and test sets are presented side by side to make the performance gap visually concrete.

## 5. Specific Questions for the Authors

1. What is the SHAP feature importance ranking when computed on the test set? If cooperation momentum and events in the prior 5 years drop in rank on the test set, this would confirm that their validation-set importance reflects endogenous overfitting rather than stable predictive relationships.

2. For the rolling-window analysis, do any windows show climate features contributing positively to QWK? If climate's contribution is window-dependent, the blanket dismissal is inappropriate.

3. The XGBoost model was tuned via 100-trial Optuna with nested 5-fold basin-grouped CV. What was the variance in QWK across the 5 folds? High fold variance would indicate that even within the training period, the model is unstable, which compounds the generalization concern.

4. The authors note that treaty formation rate increased in SHAP importance post-2000. But how many treaties were actually formed in the 2000-2008 test period? If the number is very small, the feature may be uninformative for test predictions regardless of its SHAP importance in the validation era.

5. Have the authors considered training a model on the combined train+validation set and evaluating on test? The reported experiment (test QWK improves from 0.120 to 0.181) used default hyperparameters. What happens with Optuna tuning on the combined set using a rolling-origin CV scheme?

6. The bilateral indicator ranks third in permutation importance (0.024). Is this feature potentially proxying for the number of countries (rank 1 in SHAP)? What is the correlation between these two features, and does removing one change the importance of the other?

7. For the autoregressive feature analysis in Table 4: the "with autoregressive" model achieves test QWK of 0.132, while the "without" achieves 0.146. Both are substantially below the headline test QWK of 0.290 from the Optuna-tuned model. Is this because Table 4 uses default hyperparameters? If so, what is the Optuna-tuned test QWK without autoregressive features? This is arguably the most important number in the paper and it is missing.

8. The geographic concentration finding (81.3% of conflict events in 10 basins) raises the question: what is the model's test performance if these 10 basins are excluded from training? If performance collapses, the model has essentially memorised basin-specific patterns rather than learning transferable conflict dynamics.

## 6. Summary Judgement

This paper tackles an important problem with methodological sophistication that exceeds most prior work in the field. The ordinal treatment, the ablation framework, the temporal SHAP decomposition, and the candid reporting of the validation-to-test gap are all commendable. However, the 42% generalization gap is not merely a limitation to be listed; it is the paper's most important finding and it undermines the evidentiary basis for the substantive claims about feature group importance, climate irrelevance, and institutional dynamics. The autoregressive feature endogeneity is more consequential than the current treatment suggests, the ablation protocol needs exhaustive or at least Shapley-value-based feature group evaluation, and the temporal SHAP claims require uncertainty quantification.

With the revisions outlined above, specifically re-running the headline model comparison without autoregressive features, providing all-subsets or Shapley-value feature group analysis, adding bootstrap CIs to the temporal SHAP decomposition, and recalibrating the conclusions to the test-set (not validation-set) evidence, this paper could make a strong contribution. In its current form, the conclusions are calibrated to the validation performance rather than the test performance, and the test performance tells a more sobering story that should be the paper's centre of gravity.
