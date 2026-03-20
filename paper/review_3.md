# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Journal:** *Global Environmental Change*

**Reviewer:** 3 (Machine Learning Methodology)

**Date:** 2026-03-20

---

## 1. Summary

This paper presents an ordinal machine learning pipeline for predicting the intensity of transboundary water conflict events using the TFDD dataset (6,805 events, 1948--2008). The authors apply a sequential ablation protocol to select features from eight external data sources, finding that economic indicators and temporal dynamics improve prediction while climate variables do not. The best model, an Optuna-tuned XGBoost classifier, achieves a validation QWK of 0.523 but drops to 0.293 on the held-out test set (2003--2008). SHAP temporal decomposition is used to argue that treaty formation rate, rather than treaty stock, drives conflict resolution in the post-2000 era.

---

## 2. Significance and Novelty

The paper addresses a genuine gap: prior conflict prediction work has largely ignored the ordinal structure of the BAR scale, and no prior study has systematically evaluated which external data sources contribute predictive power at the TFDD event level. The use of QWK as a primary metric is appropriate and methodologically sound for ordinal targets. The SHAP temporal decomposition across geopolitical eras is a useful analytical contribution. However, the core ML methodology (XGBoost with Optuna tuning, SHAP post-hoc explanation) is well-established and does not advance the state of the art in conflict forecasting methods. Compared to the ViEWS system, the ACLED-based PRIO-GRID literature, or the WPS Partnership's LSTM pipeline, this work operates at a substantially smaller scale and does not address the sequential, spatiotemporal structure of the data. The ablation protocol, while intuitive, is methodologically weaker than standard feature selection approaches (see Major Concern 3). The contribution is best characterised as a solid applied ML study rather than a methodological advance, which should be reflected in the framing.

---

## 3. Major Concerns

**3.1. Sample size relative to model complexity and overfitting risk.**
The dataset contains 6,805 events split into 4,271 training, 1,514 validation, and 1,005 test samples. The Optuna-tuned XGBoost model searches over 9 continuous/integer hyperparameters across 100 trials, each evaluated on the 1,514-sample validation set. With 45 features, the effective degrees of freedom of a deep boosted tree ensemble (max_depth=8, 382 trees, as reported in the best trial) are very high relative to the training set size. The neutral class contains only 273 total events (109 in training by proportion), which is critically small for any classifier to learn a meaningful decision boundary. The test-set confusion matrix confirms this: neutral-class precision is 0.036 and recall is 0.059, meaning the model essentially cannot predict the neutral class. The authors should report per-class sample sizes in each split and discuss whether the 4-class formulation is viable given the neutral class size. A 3-class formulation (merging neutral into mild cooperation, or dropping it) should be tested as a robustness check.

**3.2. The validation-to-test QWK drop (0.523 to 0.293) is severe and inadequately addressed.**
A 44% relative decline in the primary metric between validation and test sets is not merely "informative" as the authors claim; it raises fundamental questions about model validity. The manuscript attributes this to "distributional shift inherent in the 2003--2008 test period" and the Iraq War, but provides no quantitative evidence for this claim. Specifically:

- No distributional comparison (e.g., feature-space shift via MMD, PSI, or simple summary statistics) between validation and test periods is provided.
- The authors do not distinguish between temporal concept drift (the true data-generating process changed) and overfitting to the validation set (the model memorised validation-period patterns). These require different remedies.
- The claim that ablation pruning "confirms that aggressive feature selection reduces overfitting" (because the 45-feature model outperforms the 82-feature model on test by +0.069) is circular: both models show massive val-to-test degradation; the pruned model simply degrades less.
- Retraining the final model on train+validation combined before test evaluation (as done in Cell 8 of the notebook) changes the training distribution and makes the test result not directly comparable to the validation result. This is standard practice but should be noted explicitly, as it means the test model has never been evaluated on its own validation set.

The authors should (a) provide distributional shift diagnostics, (b) report test-set QWK for the model trained on training data only (without retraining on train+val) to isolate the effect of retraining from the effect of distributional shift, and (c) consider time-series cross-validation with multiple temporal folds (e.g., expanding window) to produce more robust performance estimates than a single train/val/test split.

**3.3. The ablation protocol is sequential, path-dependent, and methodologically fragile.**
The ablation adds feature groups in a fixed order (climate, governance, economic, AQUASTAT, asymmetry, temporal) and applies a greedy retain/discard rule. This design has several problems:

- **Path dependence.** If the order were reversed (e.g., temporal first, then economic, then climate), the retained set could differ. Climate variables might provide marginal value if added after temporal features are already in the model (which capture similar variance via cooperation momentum), but could be useful in a model lacking temporal features. The paper does not test alternative orderings or demonstrate order-invariance.
- **No interaction testing.** Feature groups are tested independently against the current retained set. Complementary groups that are individually weak but jointly strong (e.g., climate + governance) would be missed.
- **No statistical significance.** The retain/discard threshold appears to be delta > 0. The governance group is discarded at delta = +0.004, but this is not tested for significance. With a single validation split, QWK estimates have non-trivial variance (the bootstrap CIs for default LightGBM span ~0.09). A delta of 0.004 is well within noise.
- **Recommendation.** The authors should complement or replace the sequential ablation with a model-agnostic feature selection method: permutation importance with cross-validated significance testing, Boruta (which wraps random forests to test features against shadow features), or LASSO/elastic net as a pre-filter. At minimum, the authors should test 2--3 alternative orderings and report whether the same feature groups are retained.

**3.4. Optuna tuning on a single validation fold risks overfitting to the validation set.**
The 100-trial Optuna study directly optimises QWK on the 1,514-sample validation set. The objective function (Cell 5 and Cell 6 of the modeling notebook) trains on the training set and evaluates on the validation set in every trial, meaning the validation set functions as a second training signal. With 100 trials over a 9-dimensional hyperparameter space, the risk of overfitting to the validation set is real, particularly because:

- The search space includes high-capacity parameters (max_depth up to 10, n_estimators up to 1,000).
- The best XGBoost trial achieves max_depth=8 with 382 trees, which is high capacity for 4,271 training samples.
- The best LightGBM and XGBoost QWK values (0.517, 0.523) are suspiciously close, and both show similar degradation on test, consistent with both models overfitting to validation-set idiosyncrasies.

The proper approach is nested cross-validation: the inner loop tunes hyperparameters, and the outer loop estimates generalisation error. Alternatively, the authors should use the basin-grouped 5-fold CV (which they mention in the Methods but do not appear to use for Optuna) as the tuning objective rather than the single validation set. The authors should report the Optuna optimisation history (score vs. trial number) to assess whether 100 trials saturated the search or whether the objective was still improving, which would indicate under-exploration.

**3.5. Missing value handling is inadequate for 64% missingness in governance features.**
The EDA notebook reveals that WGI governance features have ~36% completeness (i.e., ~64% missing), Polity scores have ~60% missing, and AQUASTAT features have similarly high missingness. Median imputation applied to features with >50% missing values is known to (a) attenuate feature variance, (b) bias correlation structure, and (c) introduce artificial modes in the feature distribution. For the governance group, the median-imputed values dominate the actual observations, meaning the model is largely learning from imputed constants rather than real data. This could explain why governance features fail the ablation test: they may have been rendered uninformative by the imputation strategy itself.

The authors should:
- Report per-feature missingness rates for all 45 retained features and all 82 candidate features.
- Compare median imputation against (a) LightGBM's native missing value handling (which learns optimal split directions for missing values and is one of the model's key advantages), (b) multiple imputation (MICE), and (c) missingness indicators (binary flags). The fact that LightGBM natively handles NaN makes median imputation an unnecessarily lossy preprocessing step for tree-based models.
- Discuss whether the governance features' failure in ablation is an artefact of imputation rather than a genuine finding about governance's irrelevance to conflict prediction.

---

## 4. Minor Concerns

**4.1. Bootstrap CIs resample predictions, not models.**
The bootstrap confidence intervals (Cell 7 of the modeling notebook) resample (y_true, y_pred) pairs from the validation set and recompute QWK on each resample. This captures sampling uncertainty in the metric estimate given a fixed model, but does not capture model uncertainty (i.e., variability due to different training sets). The CIs are therefore narrower than true generalisation uncertainty, which requires retraining. This limitation should be stated. Additionally, the percentile bootstrap (2.5th, 97.5th) is used, which is known to have poor coverage for bounded metrics like kappa. The BCa (bias-corrected and accelerated) bootstrap would be more appropriate.

**4.2. Reproducibility and code/data availability.**
The manuscript does not include a data availability statement or a link to a code repository. The TFDD is publicly available, but the enriched dataset (events_enriched.parquet with 104 columns) is a derived product that requires the full pipeline to reproduce. The authors should release the enrichment pipeline, the exact feature engineering code, the Optuna study objects (or at least the best hyperparameters), and a requirements file. Without these, the ablation results and model performance are not independently verifiable.

**4.3. Comparison fairness between ordinal regression and tree models.**
The ordinal regression baselines (LogisticAT, OrdinalRidge) are trained on the scaled data with default alpha=1.0. No hyperparameter tuning is performed for these baselines, while the tree models receive 100-trial Optuna optimisation. This creates an unfair comparison. The ordinal models should receive at minimum a grid search over their regularisation parameter (alpha). Additionally, ordinal regression models assume linear decision boundaries and cannot capture interactions; a fairer comparison would include a nonlinear ordinal model such as CORN (conditional ordinal regression network) or a threshold-based neural approach.

**4.4. SHAP analysis is performed on a non-tuned model.**
The SHAP analysis in Notebook 03 trains a default XGBoost (n_estimators=500, max_depth=6, learning_rate=0.05) on the training set only, not the Optuna-tuned model that is reported as the best model in the paper. The feature importance rankings from a default model may differ from those of the tuned model (max_depth=8, 382 trees, learning_rate=0.011). The SHAP analysis should be performed on the exact model whose performance is reported, or the discrepancy should be acknowledged and justified.

**4.5. The "autoresearch" framing is unnecessary and unsupported.**
The paper cites "Karpathy, 2024" for the ablation protocol but provides no full reference. The term "autoresearch paradigm" does not appear in the published ML literature. The ablation protocol described is a standard greedy forward feature selection, which has a long history in the ML literature (e.g., Guyon & Elisseeff, 2003). The framing should be corrected to cite the appropriate methodological antecedents.

---

## 5. Questions for Authors

**Q1.** The Optuna objective function evaluates directly on the validation set (Cells 5--6 of the modeling notebook). Did you consider using the basin-grouped 5-fold cross-validation within the training set as the Optuna objective instead? This would prevent the validation set from being consumed by hyperparameter tuning, preserving it as a genuinely unseen evaluation set. Can you report what QWK the tuned models achieve under basin-grouped CV on the training set alone?

**Q2.** The SHAP temporal decomposition (Cell 7, Notebook 03) computes era-specific mean |SHAP| values from a single model trained on the full training set (pre-1996). Since the model was not trained on post-2000 data, the SHAP values for post-2000 events reflect the model's extrapolation behaviour, not necessarily the true feature-outcome relationships in that era. Have you considered training era-specific models (or at least a model trained through 2002) and comparing their SHAP profiles? How confident are you that the temporal SHAP shifts reflect genuine shifts in the data-generating process rather than model extrapolation artefacts?

**Q3.** The neutral class (BAR = 0, 4.0% of events) achieves test precision of 0.036 and recall of 0.059. This is worse than random guessing for a 4-class problem. Given that the QWK metric penalises distant misclassifications quadratically, the model may be "gaming" QWK by predicting the neutral class very rarely and concentrating predictions on the three larger classes. Can you report the class-specific prediction frequencies on the test set, and can you demonstrate that the QWK improvement from tuning is not driven entirely by the model learning to suppress neutral predictions?

---

## 6. Recommendation

**Major Revision.**

The paper addresses a relevant applied problem with a reasonable methodological approach, and the core finding that climate variables do not improve event-level prediction is potentially valuable. However, the severe validation-to-test performance drop, the path-dependent ablation protocol, the Optuna overfitting risk, the inadequate missing value treatment, and the SHAP analysis being performed on a different model than the one reported collectively undermine confidence in the results. These are addressable issues, but they require substantial additional experimentation (nested CV or multi-fold temporal CV, alternative feature selection, imputation comparison, SHAP on the correct model) that goes beyond editorial revision. I would be willing to review a revised manuscript that addresses these concerns with new experiments.
