# Meta-Review: Predicting Transboundary Water Conflict Outcomes

**Journal**: *Global Environmental Change*

**Role**: Associate Editor

**Date**: 24 March 2026

**Manuscript**: "Predicting Transboundary Water Conflict Outcomes: An Ordinal Machine Learning Benchmark on the TFDD"

**Reviews synthesized**: Reviewers 1 through 6

---

## 1. Overview

This manuscript presents an ordinal machine learning benchmark for predicting the intensity of transboundary water conflict events using the TFDD (6,805 events, 1948-2008), enriched with eight external data sources. The authors deploy a sequential ablation protocol to evaluate feature group contributions, train gradient-boosted tree classifiers with Optuna hyperparameter tuning, and apply SHAP temporal decomposition to investigate shifting driver importance across geopolitical eras. The headline findings are that economic indicators and temporal dynamics are candidate predictors of conflict intensity, that basin-averaged annual climate variables show ambiguous incremental value, and that the rate of treaty formation has grown in importance relative to treaty stock in the post-2000 era.

All six reviewers recommend **Major Revision**. There is broad agreement that the manuscript addresses a genuine gap, that the ordinal treatment of the BAR scale is a welcome advance, and that the transparency about limitations (particularly the validation-to-test performance gap and autoregressive endogeneity) sets a commendable standard. However, the reviewers converge on several substantive concerns that, taken together, require significant additional analysis before the claims can be considered reliable. I concur with the unanimous Major Revision recommendation.

The authors have clearly undertaken a substantial revision cycle before this submission, as the manuscript already addresses several concerns that reviewers would typically raise in a first round: autoregressive features are separated from the primary model, ablation robustness across orderings is reported, class distributions per split are disclosed, and the climate finding is hedged with resolution-specific language. This responsiveness to anticipated critique is noted and appreciated. The issues below therefore represent the residual concerns that remain after this self-correction.

---

## 2. Consensus Across Reviews

### 2.1 Universal or near-universal concerns (raised by 4+ reviewers)

**The validation-to-test performance gap (raised by all 6 reviewers).** Every reviewer identifies the QWK degradation from validation to test as a central concern. Reviewers 1, 3, and 4 characterize the 42-44% decline (on the autoregressive model) as undermining the evidentiary basis for substantive claims. The manuscript now reports the primary model's gap as 26% (0.403 to 0.298), which is less severe but still substantial. Reviewer 4's point is well taken: the gap is not merely a limitation to be listed but is arguably the manuscript's most important empirical result, and all interpretive claims must be calibrated to the test-set evidence rather than the validation-set evidence. The rolling-window analysis (QWK 0.145-0.280) reinforces this: the validation period may be anomalously predictable.

**Path-dependent ablation protocol (raised by Reviewers 1, 2, 3, 4, 6).** Five reviewers identify the sequential ablation as methodologically fragile due to path dependence. The manuscript now reports three orderings and acknowledges sensitivity, which is a substantial improvement. However, Reviewer 4's request for an all-subsets analysis (2^7 = 128 combinations, computationally trivial) is both feasible and would definitively resolve the question of which feature groups contribute. The manuscript's Robustness section now reports an all-subsets result (baseline + governance + asymmetry + temporal as the best subset, QWK 0.441), which directly contradicts the sequential ablation results. This finding should be promoted to much greater prominence, as it fundamentally changes the narrative about which feature groups matter.

**Overstatement of the climate null result (raised by Reviewers 1, 2, 4, 6).** Four reviewers argue that the claim about climate variables is overreaching given the spatial and temporal scale mismatch. The manuscript has been revised to hedge the claim substantially, presenting it as resolution-specific rather than substantive. However, Reviewer 6's observation that climate is retained under the reversed ordering, and that grouped permutation importance shows a positive collective climate contribution (+0.019), means the finding is genuinely ambiguous rather than null. The abstract and conclusion now reflect this ambiguity, which is appropriate.

**Autoregressive feature endogeneity (raised by Reviewers 1, 2, 4, 5).** Four reviewers raise concerns about cooperation momentum, prior event counts, and event escalation as mechanically correlated with the target. The manuscript now separates these into a secondary model and reports that removing them improves test performance, which is the correct analytical response. The primary model (42 features, non-autoregressive) should indeed be the headline result. Reviewer 4's observation that the autoregressive ablation was conducted with default hyperparameters rather than the Optuna-tuned model has been partially addressed by reporting tuned results for both feature sets.

**Thin engagement with the hydropolitics literature (raised by Reviewers 2, 5).** Reviewers 2 and 5 argue that the theoretical framing is insufficiently deep. The manuscript now cites Zeitoun and Mirumachi (2008), TWINS, Cascao (2009), Brochmann and Hensel (2009), Conca et al. (2006), and Bernauer and Bohmelt (2020), representing a substantial improvement. However, the operationalization gap persists: the manuscript invokes hydro-hegemonic mechanisms (discursive power, agenda-setting) but the features capture only material power asymmetry. The Discussion now acknowledges this gap explicitly, which is appropriate, but Reviewer 5's suggestion to include upstream/downstream position as a feature remains unaddressed and is both theoretically motivated and practically feasible from the TFDD spatial database.

### 2.2 Concerns raised by 2-3 reviewers

**BAR scale validity as a prediction target (Reviewers 2, 5).** Both argue that the BAR scale was designed as a coding instrument, not a dependent variable, and that the 4-class grouping involves theoretically underspecified discretization choices. The manuscript now includes a caveat about this in the Target Formulation section and reports sensitivity analyses with 3-class and 5-class groupings in Extended Data. The absence of inter-coder reliability statistics remains a gap.

**QWK as the sole primary metric (Reviewers 1, 3, 4).** Three reviewers note that QWK is dominated by the majority class and that macro-F1 tells a different story. The manuscript now reports both metrics throughout, which addresses the concern. The neutral class remains essentially unpredictable (precision 0.020, recall 0.206 on test), which is a sample size problem rather than a model problem, and reporting metrics with and without the neutral class would further clarify.

**Missing value handling for governance features (Reviewers 3, 4, 5, 6).** The 60-64% missingness in governance indicators, treated with median imputation, may have rendered these features uninformative before they were tested in the ablation. The manuscript now reports that native NaN handling and missingness indicators improve performance, and flags this as a priority for future work. Reviewer 3's observation that LightGBM natively handles missing values, making median imputation unnecessarily lossy, is well taken. The governance ablation should be re-run with native NaN handling before the authors can claim that governance features do not contribute.

**Non-independence of events within basins (Reviewers 2, 5).** Basin-level autocorrelation means the model may be learning basin identity rather than generalizable conflict drivers. The continent-level spatial CV (mean QWK 0.248, range 0.062-0.417) provides some evidence on this point, though basin-level holdout experiments would be more informative. The geographic concentration finding (81.3% of conflict in 10 basins) reinforces this concern.

**Reproducibility and code/data availability (Reviewers 1, 3).** The manuscript now includes a Data and Code Availability section with a GitHub repository link, which addresses this concern.

---

## 3. Top 5 Actionable Improvements Ranked by Impact on Acceptance Probability

**1. Promote the all-subsets analysis to a primary result and recalibrate the narrative accordingly.**

The all-subsets analysis identifies baseline + governance + asymmetry + temporal (QWK 0.441) as the best-performing feature group combination, which differs from all three sequential ablation orderings. This result, currently buried in the Robustness section, is the most informative and order-independent evidence about which feature groups matter. It should replace the sequential ablation as the primary feature selection result, with the sequential ablation demoted to a methodological comparison. Critically, the all-subsets winner includes governance (which the sequential ablation discarded) and temporal features (discarded under two of three orderings), fundamentally changing the story. The authors should report test QWK for the all-subsets best configuration and compare it to the sequential ablation configurations.

**2. Re-run the governance ablation with native NaN handling and report results.**

The finding that governance features were discarded during ablation may be an artefact of median imputation applied to features with 60-64% missingness. The manuscript acknowledges this but defers the fix to future work. Given that the improvement from native NaN handling (+0.020 test QWK) is comparable to the retention thresholds used in ablation, and given that the all-subsets analysis retains governance, this re-analysis is essential before any claim about governance's predictive contribution can be made. This is computationally inexpensive and should be done in this revision cycle.

**3. Report bootstrap confidence intervals on the temporal SHAP decomposition and the all-subsets feature group contributions.**

The 56.5% increase in treaty formation rate importance is a point estimate with no uncertainty quantification. All six reviewers (explicitly or implicitly) request confidence intervals on this claim. Similarly, the ablation deltas used for retention decisions lack statistical inference. Bootstrap CIs on era-specific SHAP values and on QWK differences between feature group configurations would establish which findings survive uncertainty and which are noise. The manuscript mentions that bootstrap CIs on the temporal SHAP are in Extended Data; these should be reported in the main text.

**4. Test at least one finer-grained climate specification.**

Reviewer 6 makes a compelling case that testing a single coarse climate specification and finding no signal is insufficient to conclude anything about climate-conflict linkages. The manuscript now appropriately hedges the claim, but testing one sub-annual specification (e.g., minimum monthly SPEI in the 12 months preceding each event, or maximum negative precipitation anomaly) would substantially strengthen the paper regardless of the outcome. If finer-grained climate features also fail, the negative finding becomes more convincing. If they succeed, it demonstrates the scale-mismatch interpretation. This is a single additional analysis that would transform an ambiguous finding into a definitive one.

**5. Report the continent-level spatial CV results with continent-specific discussion.**

The spatial CV reveals QWK of 0.062 for North America and 0.075 for South America, meaning the model essentially fails in these regions. The manuscript now reports these numbers in a table and discusses North America's anomalous conflict ratio, but the failure modes for South America are unexplained. Practitioners need to know which regions the model should not be applied to and why. A brief discussion of what distinguishes the failing continents (low event counts, different institutional regimes, coding density biases) would substantially improve the paper's practical utility.

---

## 4. Concerns Raised by Only One Reviewer That I Disagree With

**Reviewer 3 (Section 4.3): Ordinal regression baselines should receive Optuna-level tuning for a fair comparison.**

Reviewer 3 argues that comparing default-hyperparameter ordinal regression against 100-trial Optuna-tuned XGBoost is unfair. While this is technically correct in a model-comparison paper, this manuscript's primary contribution is not a model horse race; it is a feature evaluation and interpretability study. The ordinal regression baselines serve as sanity checks, not as serious competitors. The manuscript now reports that ordinal regression baselines perform better on the full 82-feature set, acknowledging the unfairness. Exhaustive tuning of ordinal baselines would consume page space without changing the paper's substantive conclusions, since the interpretability analysis depends on tree-based SHAP regardless of which model "wins." This concern is valid but low priority relative to the five items above.

**Reviewer 2 (Section 4.3): North American conflict ratio as a "coding artefact" requiring engagement with US-Mexico transboundary water literature.**

While Reviewer 2 raises an interesting point about whether US-Mexico water relations involve genuine coercion or institutionalized disagreement, requiring the authors to disaggregate by basin and decade and engage with Mumme (2000) and Wilder et al. (2020) would substantially expand the scope of the paper beyond its core contribution. The manuscript already provides a thoughtful discussion of this finding and correctly identifies it as a BAR scale limitation. A full re-analysis of North American hydropolitics is a separate paper. However, I agree with Reviewer 5 that the finding is "more interesting than the authors seem to realize" and a brief paragraph noting the interpretive possibilities would be sufficient.

---

## 5. Additional Points of Improvement Not Raised by Any Reviewer

**5.1. The SHAP analysis is performed on a different model than the headline result.**

Reviewer 3 notes (Section 4.4) that the SHAP analysis uses a default XGBoost, not the Optuna-tuned model. The manuscript's response to this is unclear. If the SHAP feature importance rankings are derived from a model that differs in depth, learning rate, and regularization from the model whose performance is reported, the interpretive claims may not apply to the headline model. The authors should either perform SHAP on the exact tuned model or demonstrate that the rankings are stable across model specifications (e.g., rank correlation between default and tuned SHAP importance vectors).

**5.2. The "year" feature should be replaced or supplemented with principled temporal encodings.**

Multiple reviewers flag raw year as a problematic feature (Reviewers 1, 4, 5), but none explicitly recommend a concrete alternative. The manuscript notes that era indicators (Cold War, post-2000) provide coarser but more principled temporal encoding. The authors should consider replacing raw year with decade indicators, a smooth spline of time, or the residual from a linear time trend, any of which would capture secular dynamics without preventing extrapolation. Alternatively, if year is retained for SHAP analysis, it should be excluded from the primary predictive model.

**5.3. The collinearity between treaty formation rate and cooperation momentum (r = 0.303) warrants variance inflation factor analysis.**

The manuscript reports this correlation but does not assess its impact on SHAP stability. When correlated features are present, SHAP distributes importance across them in ways that depend on the tree structure, meaning the individual SHAP rankings for these features are unreliable even if the combined importance is stable. Reporting SHAP importance for these features as a group (treaty formation rate + cooperation momentum combined) alongside the individual rankings would provide a more robust estimate.

**5.4. The class distribution shift between training and test periods is now reported but not formally tested.**

The manuscript discloses that the conflict class rises from 16.3% in training to 24.8% in test, and strong cooperation drops from 30.0% to 9.7%. This is a dramatic shift that directly explains much of the validation-to-test gap. A chi-squared test or population stability index would formalize whether this shift is statistically significant, and would help distinguish concept drift from label shift, which require different remedies.

**5.5. No discussion of how the single-author design affects reproducibility and verification.**

The manuscript is single-authored. While this is not disqualifying, the absence of independent verification of the analysis pipeline increases the importance of the code repository. The authors should ensure that the repository contains a fully executable pipeline with pinned dependency versions, not merely analysis notebooks.

---

## 6. Final Editorial Recommendation

**Major Revision.**

The manuscript addresses a genuine gap in the quantitative hydropolitics literature. The ordinal treatment of the BAR scale, the multi-source feature integration, the ablation framework with robustness testing, the separation of autoregressive features, and the candid self-assessment of limitations all reflect mature scientific thinking that exceeds the standard in this subfield. The manuscript has clearly undergone significant internal revision, as evidenced by its pre-emptive addressing of several concerns that reviewers would typically raise.

However, three issues prevent acceptance in the current form:

(a) The all-subsets analysis contradicts the sequential ablation results, and the narrative has not been updated to reflect this. The paper's structural logic still flows through the sequential ablation tables, when the all-subsets result is the more informative and order-independent evidence. This requires a reorganization of the Results section, not merely an additional paragraph.

(b) The governance ablation with native NaN handling has not been performed, meaning the claim about governance features' predictive contribution rests on an analysis known to be biased by the imputation strategy. This is a concrete, bounded analysis that should be completed in this revision cycle.

(c) The temporal SHAP claims, while now appropriately hedged, still lack the uncertainty quantification needed to distinguish signal from noise. If the bootstrap CIs on era-specific SHAP values overlap (which is plausible given unequal era sizes), the 56.5% figure must be further downgraded from a "finding" to an "observation."

I am confident that these revisions are feasible within a single revision cycle and would bring the manuscript to a publishable standard. The authors should focus on the five actionable items in Section 3 above, prioritizing items 1-3 as these require the least additional computation and have the highest impact on the paper's credibility.

---

## 7. The Single Most Impactful Structural Change

**Restructure the Results section so that the all-subsets feature group analysis is the primary evidence for feature group importance, with the sequential ablation serving as a methodological comparison rather than the narrative backbone.**

Currently, the paper's logic flows as: sequential ablation (Tables 1-2) identifies retained features, then model comparison (Table 3) evaluates models on those features, then SHAP explains the model. The sequential ablation occupies the structural position of the primary evidence despite being acknowledged as path-dependent. The all-subsets result, which is order-independent and exhaustive, is buried in the Robustness section.

Inverting this structure would: (a) make the paper's strongest evidence its most prominent, (b) resolve the concern that the headline feature groups depend on an arbitrary ordering, (c) naturally incorporate governance features (which the all-subsets analysis retains), changing the narrative from "governance does not matter" to "governance contributes in combination with asymmetry and temporal features," and (d) reframe the sequential ablation as a useful methodological cautionary tale rather than a primary result. This single change would address concerns raised by at least four reviewers while strengthening the paper's most novel contribution: demonstrating that forward selection is a noisy estimator of feature group value, and that exhaustive evaluation yields qualitatively different conclusions.

The revised Results structure would be:

1. All-subsets feature group analysis (primary evidence for which groups contribute)
2. Sequential ablation as methodological comparison (demonstrating path dependence)
3. Model comparison on the all-subsets-selected feature set
4. SHAP analysis and temporal decomposition
5. Spatial generalization

This reorganization does not require new analyses; it requires reframing existing analyses in order of their evidential weight.
