# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Reviewer 10 -- Applied Machine Learning Methodology, Model Evaluation, and Reproducible ML Pipelines**

---

## General Assessment

This manuscript presents an ordinal classification benchmark for predicting transboundary water conflict outcomes using the TFDD. The authors demonstrate commendable methodological awareness, particularly regarding autoregressive endogeneity, ablation path-dependence, and the limitations of SHAP as a causal tool. The work is among the more careful ML-for-conflict studies I have reviewed. However, several aspects of the ML pipeline require clarification or revision before the methodology can be considered fully sound. I organize my concerns into major and minor categories below.

---

## Major Concerns

**1. The nested cross-validation protocol is incompletely described and may not be truly nested.**

The manuscript states that "Optuna Bayesian optimisation" used "nested 5-fold basin-grouped CV" (Table 3), but the description in the Methods section says only that "hyperparameter tuning used basin-grouped 5-fold cross-validation within the training set." This is inner-loop CV for hyperparameter selection, which is standard, but the term "nested CV" typically implies an outer loop as well, where each outer fold produces an independent model whose performance is evaluated on the held-out outer fold. What is actually described is a single temporal train/validation/test split with inner-loop CV for tuning. The authors should either (a) clarify that this is a single-split protocol with inner CV for tuning (not nested CV in the standard sense), or (b) describe the full outer-loop protocol if one was used. As written, the terminology is misleading and overstates the rigor of the evaluation. If only a single temporal split was used for final evaluation, then the reported test QWK of 0.298 is a single point estimate with no associated variance from the evaluation protocol itself, which should be stated explicitly.

**2. Model comparison is not fair to ordinal regression baselines.**

The authors acknowledge in the robustness section that "ordinal regression baselines performed better on the full 82-feature set... than on the pruned 45-feature set, indicating that the LightGBM-optimized feature selection systematically disadvantaged linear models." This is a serious fairness issue. The ablation was conducted with LightGBM, and the retained feature set was then handed to ordinal regression models. Tree-based models handle interactions, nonlinearities, and missing values natively; linear ordinal models do not. The comparison in Table 3 is therefore biased against the ordinal baselines by design. To make the comparison fair, the authors should either (a) run ablation separately for ordinal models with their own feature selection, (b) report ordinal model performance on the full 82-feature set in the main Table 3 (not only in the robustness section), or (c) at minimum, include polynomial/interaction features and proper imputation for the ordinal models. Without this, the claim that XGBoost "outperforms ordinal regression baselines" (Abstract) rests on a rigged comparison.

**3. Bootstrap confidence intervals capture only metric estimation uncertainty, not the uncertainty that matters most.**

The authors correctly note that their bootstrap CIs "resample predictions from a fixed model, capturing metric estimation uncertainty but not model training uncertainty." This is an important acknowledgment, but the implications are not fully drawn out. The reported CIs (e.g., [0.461, 0.539] for the 45-feature model) are narrow because they reflect sampling variance in the evaluation set, not instability in the learned model. For a study that emphasizes robustness, the absence of any estimate of model training variance is a gap. The fold-level QWK variance from the inner CV is mentioned as a "complementary estimate of model instability" but is never reported. This should be reported, or alternatively, repeated train/test splits (e.g., via a rolling temporal window) should be used to estimate full-pipeline variance. The current CIs give a false sense of precision.

**4. SHAP analysis is performed on a model that includes autoregressive features, contradicting the primary model designation.**

The SHAP analysis in Results reports that "events in the prior 5 years ranked second (0.319)" and "cooperation momentum ranked third (0.318)," both autoregressive features. Yet the primary model is designated as the 42-feature non-autoregressive version. This creates a contradiction: the interpretability analysis is conducted on a model the authors themselves argue should not be trusted for generalization. The SHAP analysis should be re-run (or at least also reported) on the 42-feature primary model, and the current SHAP results should be clearly labeled as pertaining to the 45-feature model. If the 42-feature SHAP analysis produces different rankings, that difference is itself informative and should be discussed. As presented, readers may incorrectly interpret the SHAP results as explaining the primary model's behavior.

**5. Temporal SHAP confidence intervals are mentioned but not shown in the main text.**

The manuscript states that "95% bootstrap CIs [are] reported in Extended Data" for the temporal SHAP decomposition, but the Extended Data section provided does not contain these CIs. The headline claim that treaty formation rate importance increased 56.5% rests entirely on point estimates of era-specific mean absolute SHAP values. Without CIs on these era-specific estimates, the reader cannot assess whether 56.5% is distinguishable from noise. Given that the post-2000 era contains the fewest events (2000-2008, roughly 6 years), sampling variance in that era's SHAP values could be substantial. These CIs should be reported in the main text, not deferred to supplementary material, since the 56.5% figure is the headline finding.

**6. The all-subsets analysis undermines the ablation protocol but its implications are not fully confronted.**

The robustness section reports that exhaustive search over 64 subsets found a different optimal combination (baseline + governance + asymmetry + temporal, QWK 0.441) than any of the three sequential ablation orderings. This is a striking result: forward selection failed to find the best subset in all three attempts. The authors note this confirms that "forward selection is a noisy estimator of feature group value," but then proceed to use the forward-selection result (45 features) as the primary model anyway. Why not use the all-subsets winner? If computational cost is the concern, the 64-subset search was already done. If the concern is overfitting to the validation set, the same concern applies to the ablation-selected set. This needs explicit justification.

---

## Minor Concerns

**7. McNemar's test is designed for binary outcomes and is not directly applicable to four-class ordinal predictions.**

Extended Data Figure 3 reports McNemar's test for pairwise model comparison, but McNemar's test compares the discordant error patterns of two classifiers on a binary outcome. For a four-class problem, either a multiclass extension (e.g., Stuart-Maxwell test) should be used, or the authors should clarify how they binarized the predictions for McNemar's test and justify that choice.

**8. The 0.005 QWK retention threshold for ablation is arbitrary and not justified.**

The ablation protocol retains feature groups only if they improve QWK by at least 0.005. No justification is given for this threshold. Is 0.005 QWK practically meaningful? Is it statistically distinguishable from zero given the bootstrap CIs? The bootstrap CIs on QWK deltas are mentioned in the Methods ("1,000-resample bootstrap confidence intervals on QWK deltas") but never reported in Table 1. If the CI for the AQUASTAT delta of +0.005 includes zero, the retention decision is not statistically supported.

**9. Missing baselines that would strengthen the evaluation.**

Several natural baselines are absent: (a) a random forest, which is the most common model in prior TFDD/conflict studies and would anchor the comparison to existing literature; (b) an ordinal-aware gradient boosting approach (e.g., using a custom ordinal loss function in LightGBM/XGBoost rather than treating the problem as nominal multiclass); (c) a simple k-nearest-neighbors or naive Bayes baseline to establish a floor. The current comparison jumps from ordinal logistic regression to tuned gradient boosting with nothing in between, leaving a gap in the model complexity spectrum.

**10. No calibration analysis is reported.**

For a model intended to provide "probabilistic outputs with explicitly communicated uncertainty" (Discussion), no calibration analysis is presented. Reliability diagrams or expected calibration error (ECE) would indicate whether the predicted class probabilities are meaningful. This is particularly important given the severe class imbalance (neutral class at 4%) and the observed over-prediction of the neutral class (343 predicted vs. 34 actual).

**11. The ablation uses LightGBM but the final model is XGBoost.**

The ablation protocol uses LightGBM with default hyperparameters, but the final model reported in Table 3 is XGBoost (both default and Optuna-tuned). While both are gradient-boosted tree ensembles, they differ in tree-building strategy, handling of categorical features, and regularization defaults. Feature groups retained by LightGBM ablation may not be optimal for XGBoost. At minimum, the authors should report whether the ablation decisions change when XGBoost is used as the ablation model.

**12. The Optuna search space is not reported.**

The manuscript states that 100-trial Bayesian optimization was used, and Extended Data Figure 2 shows convergence, but the hyperparameter search space (which parameters, their ranges, and the prior distributions) is not specified. Reproducibility requires this information. The number of hyperparameters being tuned relative to the number of inner CV folds also determines the risk of overfitting the validation metric.

**13. Quadratic weighted kappa assumes equal spacing between ordinal classes.**

The authors acknowledge that the BAR scale's ordinal spacing is imperfect, but the use of QWK goes further: it assumes that the four ordinal classes are equally spaced (classes 0, 1, 2, 3 with quadratic penalties on distance). A confusion between conflict (class 0) and strong cooperation (class 3) receives 9x the penalty of an adjacent-class error, regardless of whether the political consequences scale quadratically. Custom cost matrices reflecting domain-specific misclassification costs (e.g., failing to predict conflict is worse than misclassifying cooperation intensity) would be more appropriate and should at least be discussed.

**14. The temporal split may confound regime change with distribution shift.**

The class distribution shifts substantially between periods (conflict rises from 16.3% to 24.8%; strong cooperation drops from 30.0% to 9.7%). The validation-to-test performance gap could partly reflect this prior shift rather than a failure of learned patterns to generalize. A simple prior-adjusted baseline (predicting test-period class proportions) would help disentangle these effects.

**15. Reproducibility gaps.**

While code is stated to be available, several pipeline details are missing: the random seed(s) used, whether results are averaged over multiple seeds, the exact Optuna sampler (TPE, CMA-ES, etc.), and whether early stopping was used during XGBoost training. For a study positioning itself as a benchmark, full reproducibility requires these details.

---

## Summary Recommendation

The manuscript makes a genuine contribution by bringing ordinal-aware evaluation, ablation analysis, and autoregressive endogeneity diagnosis to a domain that has lacked methodological rigor. However, the ML methodology has several issues that need addressing before publication: the SHAP analysis is run on the wrong model, the model comparison is unfair to ordinal baselines, the nested CV terminology is misleading, and the all-subsets result contradicts the ablation protocol without adequate resolution. I recommend major revision with particular attention to concerns 1 through 6, which affect the validity of the paper's central claims.

**Recommendation: Major Revision**
