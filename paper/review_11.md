# Peer Review 11: Statistical Inference and Causal Identification

**Reviewer expertise**: Statistical inference, causal identification, prediction-explanation distinction in social science

**Manuscript**: "Predicting Transboundary Water Conflict Outcomes: An Ordinal Machine Learning Benchmark on the TFDD"

---

## General Assessment

This manuscript presents an ordinal machine learning benchmark for predicting transboundary water conflict outcomes using the TFDD. The author has made a commendable and largely successful effort to navigate the treacherous boundary between prediction and explanation, a boundary that most applied ML papers in political science either ignore or handle superficially. The SHAP epistemological disclaimer in the Discussion is among the most thorough I have seen in a comparable manuscript, and the "associational hypothesis" framing in the Conclusion is well-calibrated. However, several inferential claims still exceed what the evidence licenses, the endogeneity analysis conflates distinct statistical concepts, and the ablation framework has unacknowledged statistical limitations that undermine the confidence with which feature group conclusions are presented. I organise my concerns into major issues requiring revision and minor issues that would strengthen the paper.

---

## Major Concerns

**1. The endogeneity analysis conflates mechanical correlation with endogeneity proper.**

The paper identifies cooperation momentum, events in the prior 5 years, and event escalation as "autoregressive" features carrying "endogeneity risk" (Section 3.2, Table 3). This language conflates two distinct problems. Endogeneity in the econometric sense refers to a situation where an explanatory variable is correlated with the error term due to omitted variables, simultaneity, or measurement error. What the paper actually demonstrates is something different: mechanical correlation, where features derived from the same measurement instrument (BAR scale) used to construct the target variable will trivially inflate apparent predictive performance. This is a feature leakage problem, not an endogeneity problem. The distinction matters because "endogeneity" implies a causal modelling framework the paper explicitly disclaims, whereas "mechanical correlation" or "target leakage" is the correct diagnosis within a predictive modelling framework. The fact that validation QWK rises by 0.10 while test QWK falls slightly is consistent with overfitting to a leaked signal, not with endogeneity in any formal sense. I recommend replacing "endogeneity" with "target leakage" or "mechanical correlation" throughout, and clarifying that the autoregressive features are problematic because they encode information from the same coding process that generated the labels, not because of reverse causation.

**2. The ablation delta threshold of 0.005 QWK lacks statistical justification.**

The retention threshold of delta QWK >= 0.005 is stated as a design choice but never justified statistically. On a validation set of approximately 1,500 events (training pre-1996, validation 1996-2002), the sampling variability of QWK is substantial. The paper reports 1,000-resample bootstrap CIs for model-level QWK (Table 3) but does not report bootstrap CIs on the delta QWK values that drive retention decisions (Table 1). This is a critical omission. A delta of +0.005 could easily fall within the noise band of the metric. Without confidence intervals on the deltas themselves, the entire ablation protocol is vulnerable to the charge that retention decisions reflect stochastic variation rather than genuine signal. The paper partially acknowledges this by testing three orderings (Table 2), but the path-dependence finding actually strengthens the concern: if the same feature group is retained under one ordering and discarded under another, the delta is almost certainly within the noise band. The authors should compute bootstrap CIs on each delta QWK in Table 1, report how many of the seven retention decisions have CIs excluding zero, and discuss the implications for the feature group conclusions.

**3. Bootstrap confidence intervals capture the wrong source of uncertainty.**

The paper correctly notes (Methods, Section 2.6) that bootstrap CIs on validation QWK "resample predictions from a fixed model, capturing metric estimation uncertainty but not model training uncertainty." This is a significant limitation that deserves more prominence. The CIs in Table 3 tell us how precisely we can estimate QWK for a specific trained model on a specific dataset, but they do not capture the variability that would arise from retraining on different samples. In a regime where the validation-to-test gap is 26%, the dominant uncertainty is clearly model instability and distributional shift, not metric estimation noise. The fold-level variance from nested CV is mentioned as "a complementary estimate" but never reported numerically. The paper should either report fold-level QWK variance (standard deviation across the 5 inner folds) or acknowledge more explicitly that the reported CIs understate total uncertainty by an unknown but likely large factor.

**4. The SHAP temporal decomposition lacks a formal test for era differences.**

The headline finding that treaty formation rate increased 56.5% in SHAP importance from the Cold War to the post-2000 era is presented as a meaningful shift, but no statistical test is provided to establish that this difference is distinguishable from sampling noise. SHAP values are computed on subsets of different sizes (Cold War era likely contains far more events than the post-2000 era, given the 2003-2008 test cutoff). Era-specific mean absolute SHAP values will therefore have different standard errors, and a 56.5% change could be within the joint confidence band. The paper mentions "95% bootstrap CIs reported in Extended Data" for the temporal SHAP decomposition, but these are not shown in the submitted manuscript. Either include the bootstrap CIs in the main text or at minimum report whether the era-specific SHAP value distributions have non-overlapping confidence intervals. Without this, the temporal decomposition is suggestive but not demonstrated.

**5. Spatial CV analysis is underpowered and uses an inappropriate unit of grouping.**

The continent-level leave-one-out CV (Table 5) uses only 5 folds, one per continent. With k=5 and QWK values ranging from 0.062 to 0.417, the mean of 0.248 has a standard error that cannot be reliably estimated from 5 observations. More fundamentally, continents are heterogeneous aggregations. Europe includes the Danube, Kura-Araks, and Rhine basins with very different conflict dynamics. North America contains only 183 events, meaning the QWK estimate for that fold is itself highly uncertain. The paper acknowledges geographic variation but still reports the mean QWK of 0.248 as though it is a stable estimate. I would recommend either (a) using basin-level or sub-regional groupings for spatial CV to increase the number of folds and reduce within-group heterogeneity, or (b) framing the continent-level results as descriptive rather than as a formal spatial generalization test.

---

## Minor Concerns

**6. The prediction-explanation distinction is well-handled but inconsistently applied.**

The Discussion opens with an exemplary epistemological caveat about SHAP values measuring marginal feature contributions rather than causal effects. However, the language occasionally slips. For instance, "treaty formation rate increased 56.5% in importance" (Results, Section 3.4) is stated without qualification before the Discussion caveat appears. Similarly, "cooperation momentum decreased 26.0% in importance post-2000, suggesting that historical path dependence erodes" (Section 3.4) moves from a SHAP statistic to a substantive claim about a real-world process without the "associational hypothesis" framing that the Discussion later establishes. The Results section should either forward-reference the epistemological caveat or apply hedging language consistently from the first mention of SHAP-derived claims.

**7. McNemar's test is applied without correction for multiple comparisons.**

Extended Data Figure 3 reports a pairwise p-value matrix for all model pairs using McNemar's test, but the caption states "values shown are raw p-values." With 6 models, there are 15 pairwise comparisons. Without Bonferroni, Holm, or FDR correction, some nominally significant differences may be spurious. This should be addressed, even if only in a supplementary note.

**8. The collinearity between treaty formation rate and cooperation momentum (r = 0.303) is acknowledged but its implications are understated.**

A Pearson r of 0.303 is moderate, but in a tree-based model, even moderate collinearity can cause SHAP values to be arbitrarily split between correlated features. The paper notes that "their SHAP importance estimates share explained variance" but does not quantify how the temporal decomposition findings would change if one of the two features were removed. Given that the headline finding concerns treaty formation rate specifically, a sensitivity analysis dropping cooperation momentum (or vice versa) would substantially strengthen the temporal decomposition claim.

**9. The QWK metric imposes an implicit assumption about ordinal spacing.**

The paper acknowledges (Section 2.3) that QWK "implicitly weights ordinal distance" and that this assumption is "imperfect." This is correct but the implications are underexplored. The quadratic weighting in QWK means that misclassifying a conflict event (class 0) as strong cooperation (class 3) is penalised 9 times more than misclassifying it as neutral (class 1). Whether this weighting reflects the policy-relevant loss function is unstated. If the primary concern is conflict early warning, a metric that heavily penalises missed conflicts (e.g., class-weighted recall for the conflict class) might be more appropriate than QWK. The paper should briefly justify why QWK is the right primary metric for the stated policy motivation, or acknowledge that alternative loss functions could yield different model rankings.

**10. The "associational hypothesis" framing in the Conclusion is appropriate but could be strengthened with a DAG.**

The Conclusion presents four policy insights as "associational hypotheses requiring causal confirmation." This is exactly the right framing. However, the paper would benefit from a simple directed acyclic graph (DAG) illustrating the hypothesised causal structure, particularly showing where treaty formation rate, cooperation momentum, economic capacity, and climate sit relative to the BAR outcome. A DAG would make explicit which confounders would need to be addressed for causal identification and would help readers evaluate which instrumental variable or natural experiment strategies might be feasible. This would elevate the paper from a well-caveated predictive exercise to a genuine contribution to the causal inference discussion in hydropolitics.

**11. The 42-feature vs. 45-feature test QWK difference (0.298 vs. 0.290) may not be statistically significant.**

The paper treats the 0.008 QWK difference between the autoregressive and non-autoregressive models on the test set as evidence that "autoregressive features do not improve test generalization." On a test set of approximately 1,005 events, this difference is likely within the noise band of QWK estimation. A bootstrap test or permutation test on the test-set QWK difference should be reported to support this claim. If the difference is not significant, the argument for excluding autoregressive features must rest on the mechanical correlation argument alone (which is itself sufficient) rather than on a claimed performance comparison.

**12. The rolling-window analysis is mentioned but not shown.**

Section 4.4 references "rolling-window analysis (QWK 0.145 to 0.280 across windows)" but this analysis does not appear in either the main text or Extended Data. If this result supports the geopolitical regime dependence argument, it should be fully reported with window sizes, overlap, and confidence intervals.

---

## Summary Recommendation

The manuscript makes a genuine contribution to both the hydropolitics and applied ML literatures. The prediction-explanation distinction is handled with unusual care, and the "associational hypothesis" framing is a model that other applied ML papers in political science should emulate. However, several statistical claims need tightening. The most consequential issues are (a) the lack of statistical tests on ablation deltas, which means retention decisions may reflect noise; (b) the absence of formal tests for the temporal SHAP decomposition, which is the headline finding; and (c) the conflation of mechanical correlation with endogeneity, which muddies an otherwise clear methodological contribution. I recommend major revision with particular attention to the five major concerns above. The core results and framing are sound; the statistical apparatus supporting the inferential claims needs strengthening to match the epistemological sophistication of the prose.

**Recommendation**: Major revision.
