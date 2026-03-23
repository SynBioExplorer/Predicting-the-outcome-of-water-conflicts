# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Reviewer 5 -- Expertise: Hydropolitics, International Water Law, Environmental Governance, Political Economy of Transboundary Resources**

---

## 1. Summary

This paper presents an ordinal machine learning benchmark for predicting transboundary water conflict outcomes using the Transboundary Freshwater Dispute Database (TFDD). The authors integrate eight external data sources, apply a sequential ablation protocol to evaluate feature group contributions, and deploy SHAP-based temporal decomposition to examine how predictor importance shifts across geopolitical eras. The best model (XGBoost, nested CV) achieves a validation QWK of 0.502 but degrades substantially on the held-out test set (QWK = 0.290), and the authors identify treaty formation rate as an increasingly important predictor in the post-2000 period. The paper is technically competent and refreshingly honest about its limitations, but raises several concerns regarding theoretical framing, target variable validity, and the interpretation of key findings that require attention before publication.

---

## 2. Overall Assessment

**Recommendation: Major Revision**

The manuscript tackles a genuinely important gap: the absence of ordinal-aware, interpretable ML benchmarks for transboundary water conflict prediction. The technical execution is largely sound, the ablation robustness analysis is a welcome methodological contribution, and the authors demonstrate unusual candour about performance limitations and endogeneity concerns. However, the paper suffers from an incomplete engagement with the hydropolitics literature it invokes, a problematic operationalisation of the BAR scale as a prediction target, and insufficiently developed policy implications. The treaty formation rate finding, while interesting, is entangled in endogeneity that the authors acknowledge but do not adequately resolve. These issues are addressable through revision, but they are substantive enough to require a major revision cycle.

---

## 3. Major Concerns

**3.1. Shallow engagement with the hydro-hegemony and TWINS frameworks despite extensive citation**

The introduction provides an impressively thorough literature review, citing Zeitoun and Warner (2006), Zeitoun and Mirumachi (2008), Mirumachi and Allan (2007), Cascao (2009), and others. However, these frameworks are invoked descriptively rather than operationalised analytically. The hydro-hegemony framework makes specific claims about the mechanisms through which power asymmetry operates: control over discourse, control over technical knowledge, and the ability to set negotiation agendas. The paper reduces this rich theoretical architecture to "asymmetry ratios" (GDP ratio, dam ratio, withdrawal ratio). This is a significant conceptual flattening. Where are the measures of discursive power, agenda-setting capacity, or the "sanctioned discourse" that Zeitoun and Warner argue is central to hegemonic control?

Similarly, the TWINS framework (Mirumachi and Allan, 2007) argues that conflict and cooperation coexist simultaneously within basins at different intensities. This insight has direct implications for the prediction target: a single BAR score per event may be capturing one dimension of a multi-dimensional interaction space. The authors note this point in passing but do not address its consequences for model design. If TWINS is taken seriously, the prediction task may be fundamentally misspecified as a single ordinal outcome per event.

**Request**: The authors should either (a) operationalise hydro-hegemonic mechanisms more faithfully, perhaps using diplomatic discourse analysis proxies, institutional participation metrics, or upstream position indicators, or (b) explicitly acknowledge that the available features capture only material power asymmetry and not the broader hegemonic mechanisms that theory identifies as decisive. The TWINS implications for the single-outcome prediction framing should be discussed substantively.

**3.2. The BAR scale as a prediction target: unexamined validity concerns**

The BAR scale is treated as an unproblematic ordinal measure, but the hydropolitics community has raised significant concerns about its construct validity that this paper does not address. Several issues deserve attention:

First, the BAR scale was designed as a descriptive coding instrument, not a dependent variable for predictive modelling. Its ordinal structure assumes equal conceptual spacing between adjacent values, an assumption that is violated in practice. The distance between a BAR score of -1 (mild verbal hostility) and -2 (official diplomatic protest) is not equivalent in political consequence to the distance between -5 (small-scale military action) and -6 (extensive war). The use of quadratic weighted kappa as a metric implicitly assumes ordinal spacing, but the paper does not discuss whether this assumption is warranted.

Second, inter-coder reliability for BAR scoring is not reported. The TFDD has undergone multiple coding updates, and the consistency of BAR assignments across different coders and time periods is a known concern in the literature. If the target variable contains substantial measurement noise, a QWK ceiling well below 1.0 would be expected regardless of feature quality.

Third, the 4-class grouping (conflict, neutral, mild cooperation, strong cooperation) involves discretisation choices that the authors describe as "grounded in the original Yoffe et al. coding framework," but the specific threshold at BAR = 3 between mild and strong cooperation is not universally accepted. The sensitivity analyses mentioned (3-class, 5-class) are relegated to Extended Data, but the choice of class boundaries is consequential enough to warrant main-text discussion.

**Request**: Discuss the measurement properties and known limitations of the BAR scale as a prediction target. Report or cite inter-coder reliability statistics. Justify the 4-class grouping with reference to the substantive meaning of each boundary, not just coding conventions.

**3.3. Treaty formation rate endogeneity is more severe than acknowledged**

The authors deserve credit for flagging the conceptual circularity of treaty formation rate as a predictor, noting that treaties are themselves coded as cooperative events in the TFDD. However, the treatment of this problem is insufficient. The concern is not merely that treaty-making and cooperative BAR outcomes share a "latent cooperativeness." The problem is structural: in the TFDD coding framework, the signing of a treaty is itself an event that receives a cooperative BAR score. A basin that signs three treaties in the preceding five years will therefore have (a) a high treaty formation rate, (b) a higher cooperation momentum from those treaty-signing events, and (c) a subsequent event that is more likely cooperative because it occurs in a basin where cooperative dynamics are already dominant. The predictor and the target are not merely correlated through a latent variable; they are mechanically coupled through the database structure.

The temporal ordering safeguard (counting treaties strictly prior to each event) is necessary but not sufficient. If Basin X signs a treaty in 2001 (coded as BAR +6), and the next event in Basin X in 2002 is the prediction target, the treaty formation rate feature has already absorbed the 2001 cooperative signal, and the 2002 event is likely cooperative because it occurs in the same cooperative institutional trajectory. The SHAP finding that treaty formation rate importance increased 56.5% post-2000 could therefore reflect not a genuine causal mechanism but an intensification of this mechanical coupling during a period of accelerated treaty-making (the post-Cold War "treaty boom").

**Request**: The authors should (a) quantify the correlation between treaty formation rate and cooperation momentum to establish the degree of collinearity, (b) test model performance with treaty formation rate removed entirely (not just autoregressive features), (c) discuss instrumental variable approaches more concretely rather than deferring them to future work, and (d) temper the claim that this finding provides "direct empirical support" for Wolf et al.'s institutional-change hypothesis given the endogeneity concern.

**3.4. The North America conflict ratio finding demands deeper analysis**

The finding that North America has the highest continental conflict ratio (37.7%) is presented as a "limitation of the BAR scale for cross-regional comparison," but this framing is too dismissive. This result is genuinely important for the field and deserves sustained analysis rather than a parenthetical qualification.

Several interpretive possibilities exist. First, the BAR scale may indeed fail to distinguish institutionalised disagreement within robust treaty regimes from genuine inter-state hostility, as the authors suggest. If true, this has profound implications for the entire prediction task: the model would be predicting a coding artefact rather than a meaningful political phenomenon. Second, the result may reflect the fact that US-Mexico water relations involve genuine coercion, structural power asymmetry, and distributive injustice (Mumme, 2000; Wilder et al., 2020) that are not reducible to "institutionalised disagreement." The US has historically exercised water hegemony over Mexico through the 1944 Treaty framework in ways that BAR coding may legitimately capture as conflictual. Third, the result may reflect coding density: North American basins may be more thoroughly coded in the TFDD, with minor disagreements captured that go unrecorded in less-documented basins.

**Request**: Disaggregate the North American conflict ratio by basin and by decade. Report the BAR score distribution for US-Mexico and US-Canada events separately. Discuss whether the high conflict ratio reflects coding artefact, genuine hydro-hegemonic dynamics, or documentation bias. Engage with the US-Mexico transboundary water literature (Mumme, 2000; Wilder et al., 2020; Sanchez-Munguia, 2011).

**3.5. Policy implications are underdeveloped and occasionally overreaching**

The Discussion contains one explicit policy recommendation: that international organisations should "prioritise mechanisms that accelerate institutional adaptation... over the accumulation of comprehensive but rigid treaty instruments." This recommendation is insufficiently supported by the analysis and potentially misleading.

First, the finding that treaty formation rate matters more than treaty stock does not imply that quantity of treaty-making is preferable to quality. Rapid treaty formation can reflect superficial diplomatic activity (e.g., non-binding memoranda of understanding) that fails to address underlying resource allocation disputes. Bernauer and Bohmelt (2020), whom the authors cite, explicitly argue that institutional design characteristics matter more than institutional quantity, which is the opposite conclusion from the one drawn here.

Second, the recommendation to favour "flexible allocation frameworks" over "comprehensive but rigid treaty instruments" is a normative position embedded in the adaptive governance literature that the data analysis does not directly support. The model shows that treaty formation rate correlates with cooperative outcomes; it does not show that flexible frameworks outperform rigid ones.

Third, the paper does not discuss the political economy constraints on "accelerating institutional adaptation." Treaty formation is not a policy lever that can be pulled at will; it depends on political will, power relations, and negotiation capacity that are themselves shaped by the same structural factors the model identifies as predictors.

**Request**: Revise the policy discussion to (a) distinguish between treaty formation rate as a predictor and as a policy target, (b) acknowledge the political economy constraints on institutional acceleration, and (c) present policy implications as hypotheses warranting further investigation rather than operational recommendations.

---

## 4. Minor Concerns

**4.1.** The claim that this is "the first ordinal-aware machine learning benchmark on the TFDD" should be verified more carefully. While I am not aware of a direct precedent, the claim should be qualified with "to our knowledge" and the authors should confirm that Ge et al. (2022), who used boosted regression trees on TFDD-derived data, did not use ordinal loss functions or evaluation metrics.

**4.2.** The temporal split (train pre-1996, validate 1996-2002, test 2003-2008) is sensible, but the rationale for these specific boundaries is not discussed. Do they correspond to geopolitical regime changes, or are they arbitrary? The post-2003 period coincides with the Iraq War, which profoundly affected Tigris-Euphrates basin dynamics. If the test period is dominated by one geopolitical shock, the validation-to-test gap may reflect event specificity rather than general temporal non-stationarity.

**4.3.** The rolling-window analysis (paragraph in Discussion) suggests the validation period may be "unusually predictable." This is an important finding that undermines the headline QWK of 0.502. The range 0.145-0.280 across windows should be reported more prominently, perhaps as a more representative estimate of expected performance.

**4.4.** The paper uses "conflict" to refer to BAR < 0 events throughout, but in hydropolitics the term "conflict" carries specific connotations of organised violence. Many BAR < 0 events are diplomatic disputes, verbal hostility, or economic sanctions, not armed conflict. The terminology should be clarified, perhaps using "negative interactions" or "non-cooperative events" instead.

**4.5.** The 60-64% missingness in governance features (Polity V, WGI) is concerning. The finding that native NaN handling and missingness indicators improve performance by +0.020-0.022 over median imputation suggests that the missingness pattern itself is informative (i.e., the data are not missing at random). This is a common issue with governance indicators for non-OECD countries, and it deserves explicit discussion. If missingness correlates with state fragility, the imputed values may mask a genuine signal.

**4.6.** The SPEI-3 index captures 3-month drought conditions. The paper tests this at annual aggregation, which averages out the seasonal signal. A more appropriate climate specification might retain the maximum SPEI severity within the year (i.e., the driest 3-month period) rather than the annual mean. This should be noted as a limitation of the climate variable specification.

**4.7.** The continent-level leave-one-group-out cross-validation (mean QWK 0.248, range 0.062-0.417) is a useful diagnostic, but continent is a very coarse spatial grouping. The range of 0.062-0.417 suggests the model essentially fails for at least one continent. Which continent(s) yield QWK near 0.062? This should be reported, as it identifies regions where the model should not be applied.

**4.8.** Figure references (Figs. 1-4) are mentioned but the figures themselves are not included in this manuscript file. If these are in supplementary materials, the key results should be self-contained in the main text. The confusion matrix and SHAP summary plot are essential for evaluating the claims.

---

## 5. Specific Questions for the Authors

**Q1.** The hydro-hegemony framework identifies upstream/downstream position as a critical determinant of bargaining power. Did you test whether upstream/downstream positioning of event participants (derivable from the TFDD spatial database and river network topology) improves prediction? If not, this is a notable omission given the theoretical framework invoked.

**Q2.** Cooperation momentum is the third-ranked SHAP feature, but Table 4 shows that removing autoregressive features actually improves test performance. How do you reconcile a feature being ranked third in importance by SHAP while its removal improves out-of-sample generalisation? Does this indicate that SHAP importance computed on validation data is misleading when features have different in-sample vs. out-of-sample utility?

**Q3.** The post-2000 era contains only 8 years of data (2000-2008). Is the SHAP temporal decomposition stable with so few years, particularly given that TFDD event density varies substantially across years? Did you perform bootstrap resampling within each era to assess the stability of the era-specific SHAP values?

**Q4.** You note that the TFDD ends at 2008. The period since then has seen the Grand Ethiopian Renaissance Dam dispute, the Rogun Dam controversy in Central Asia, accelerating Mekong mainstream dam construction, and the securitisation of water in the context of the Syrian civil war. Do you have plans to extend the analysis using the TFDD's more recent releases or supplementary datasets? Without post-2008 validation, the practical relevance of the model is limited.

**Q5.** The BAR scale includes positive values up to +7 (voluntary unification into one nation). How many events in the dataset have BAR scores above +5? If the upper end of the scale is essentially unpopulated, the "strong cooperation" class may be dominated by BAR 4-5 events (joint commissions, multilateral accords), and the ordinal structure above that level is not being tested. This would affect the interpretation of the QWK metric.

**Q6.** You cite Warner and Zawahri (2012) on power asymmetries shaping compliance patterns. Did you consider incorporating treaty compliance or violation data as features? The TFDD itself contains some information on treaty implementation, and non-compliance events could provide a more direct measure of institutional effectiveness than treaty formation rate.

**Q7.** The bilateral indicator ranks high in permutation importance. Is this capturing a genuine substantive effect (bilateral negotiations have different dynamics than multilateral ones) or is it confounded with the number of countries feature? What is the correlation between these two variables?

**Q8.** The paper frames the 42% validation-to-test degradation as evidence that "event-level water conflict prediction at this resolution remains a fundamentally difficult task." An alternative interpretation is that the model has overfit to the training/validation period, possibly through implicit temporal leakage in features like year of occurrence. Given that raw year is the fourth-ranked SHAP feature, have you tested model performance with year excluded entirely? A model that relies on year as a top predictor is, in effect, a lookup table for secular trends rather than a generalisable predictive model.

---

## Summary Evaluation

This is a technically proficient paper that addresses a real gap in the quantitative hydropolitics literature. The ablation robustness analysis and the transparent reporting of performance limitations set a good standard for the field. However, the theoretical engagement remains surface-level despite extensive citation, the BAR scale's validity as a prediction target is unexamined, the treaty formation rate finding is entangled in structural endogeneity, and the policy implications need grounding. The North America finding is more interesting than the authors seem to realise and deserves proper investigation. With substantive revision addressing these concerns, the paper could make a valuable contribution to the growing literature on quantitative conflict prediction in environmental governance.

I would support publication after a major revision that addresses the issues above, particularly concerns 3.1, 3.3, and 3.5.
