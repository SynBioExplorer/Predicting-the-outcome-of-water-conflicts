# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Journal**: *Global Environmental Change*

**Reviewer**: Reviewer 2

**Date**: 20 March 2026

**Manuscript**: "Predicting Transboundary Water Conflict Outcomes: Ordinal Machine Learning Reveals Economic and Temporal Drivers Outweigh Climate Signals"

---

## 1. Summary

This paper applies ordinal-aware gradient-boosted tree models (LightGBM and XGBoost) to 6,805 events from the Transboundary Freshwater Dispute Database (TFDD), grouping the BAR scale into four ordinal classes and using a systematic ablation protocol to evaluate predictive contributions from eight external data sources. The authors report that economic indicators and temporal dynamics (particularly treaty formation rate) are the strongest predictors of conflict intensity, while climate variables provide no predictive gain. SHAP-based temporal decomposition is used to argue that the rate of institutional change has grown in importance over time, supporting the institutional-change hypothesis of Wolf et al. (2003).

---

## 2. Significance and Novelty

The paper attempts to bridge a real gap between the qualitative richness of hydropolitics scholarship and the quantitative prediction literature. The systematic ablation protocol is a methodological contribution worth noting, and the use of ordinal classification rather than binary conflict/cooperation coding is a welcome departure from the oversimplified framing that plagues much of the quantitative environmental security literature. However, I am not convinced that the substantive conclusions move the field forward. That economic capacity structures bargaining outcomes is a central tenet of hydro-hegemony theory (Zeitoun and Warner, 2006; Cascao, 2008; Zeitoun and Mirumachi, 2008), and that institutional dynamism matters more than institutional stock is precisely the argument Wolf et al. (2003) and Giordano and Wolf (2003) made two decades ago. The ML apparatus here is confirming, with considerable computational effort, what the field already accepts on strong theoretical and case-study grounds. The paper would benefit from engaging more seriously with what its quantitative confirmation adds beyond "we ran a model and the known things came out," and from identifying genuinely novel empirical patterns that could not have been derived from existing theory.

---

## 3. Major Concerns

**3.1. Thin engagement with the hydropolitics literature.**
The literature review cites Wolf et al. (2003), Zeitoun and Warner (2006), and a handful of recent quantitative papers, but it entirely omits the substantial body of work that sits between these anchors. There is no engagement with Zeitoun and Mirumachi's (2008) Framework of Hydro-Hegemony, which introduced the continuum between conflict and cooperation that the authors' ordinal scheme implicitly invokes. Cascao's (2008; 2009) work on counter-hegemonic strategies, Mirumachi and Allan's (2007) Transboundary Waters Interaction Nexus (TWINS), Conca's (2005) analysis of treaty design and effectiveness, Giordano and Wolf's (2003) work on institutional resilience, Dinar et al.'s (2015) scarcity-institutions nexus, and Bernauer and Bohmelt's (2020) work on institutional design for transboundary rivers are all directly relevant. The paper reads as if the hydropolitics literature consists of Wolf (2003) and Zeitoun/Warner (2006) plus a few recent ML papers. For a submission to *Global Environmental Change*, this is insufficient. The theoretical framing needs to demonstrate awareness of the intervening 20 years of scholarship.

**3.2. The BAR scale was not designed as a prediction target, and the 4-class grouping is theoretically underspecified.**
The BAR scale was developed as a coding instrument for the TFDD, designed to characterize the intensity of discrete interaction events on a continuum anchored by ideal-type extremes (-7 to +7). It was not designed as, nor validated as, an outcome variable for predictive modelling. Collapsing it into four ordinal classes (conflict, neutral, mild cooperation, strong cooperation) involves substantive choices that the authors describe as "substantively meaningful thresholds" but do not defend with reference to the coding literature. Why is the neutral class a single value (BAR = 0) representing only 4% of events? The Wolf/Yoffe/Giordano coding manual treats the scale as continuous with ordinal properties; the boundaries between "mild" and "strong" cooperation at BAR = 3 are arbitrary unless justified by the event typology underlying the coding scheme. The authors acknowledge in the limitations that "alternative groupings may yield different results" but this is not merely a robustness check issue; it is a fundamental construct validity question. If the grouping boundaries shift the class distribution and alter which features are predictive, then the substantive conclusions are contingent on a discretisation choice with no theoretical anchor.

**3.3. Non-independence of events within basins and interaction sequences.**
The authors acknowledge that "events within the same basin are not independent" but understate the severity of this problem. TFDD events are frequently coded from the same diplomatic interaction, the same riparian negotiation round, or even the same news report. Events in the Jordan basin (which alone contributes a large share of conflict events) are embedded in a decades-long interaction sequence where each event is conditioned on the previous one. The authors use basin-grouped cross-validation during tuning but a simple temporal split for evaluation. This means the validation and test sets contain events from basins that are heavily represented in training, with highly autocorrelated BAR trajectories. The top feature in the SHAP analysis, "number of countries involved in the event," is essentially a basin-type proxy (bilateral vs. multilateral). The second feature, "events in the prior 5 years," and the third, "cooperation momentum," are by construction serially correlated with the target. The model may be learning basin identity and autoregressive dynamics rather than generalisable conflict drivers. A proper test would evaluate on basins entirely excluded from training (spatial cross-validation), or at minimum report basin-stratified performance metrics.

**3.4. The "climate doesn't matter" finding conflates levels of analysis and temporal scales.**
The authors claim that "climate variables provide no predictive gain at the event level" and use this to "challenge climate-centric conflict narratives." This is an overstatement that conflates two distinct analytical questions. Climate in hydropolitical theory operates as a slow-moving structural stressor that alters the resource base over which bargaining occurs (Allan, 2001; Gleditsch, 2012; Ide, 2015). It is not theorized as a proximate trigger of individual diplomatic events. Testing whether basin-averaged annual precipitation predicts discrete BAR-coded events and finding that it does not is not evidence against the role of climate in water conflict; it is evidence that the wrong climate variables are being tested at the wrong temporal and spatial resolution for the question being asked. The authors partially acknowledge this in the Discussion ("climate signals are likely too spatially and temporally diffuse to capture the proximate triggers"), but the Abstract, Introduction, and Conclusion all claim a strong finding that climate "does not matter." This framing is misleading and risks being cited in policy contexts to downplay climate adaptation investment. The paper by Ge et al. (2022) found climate sensitivity at the basin-year level; Ide et al. (2020) showed that climate effects are mediated through livelihood impacts and governance quality. The correct conclusion from the ablation is not "climate doesn't predict conflict" but "annual basin-averaged climate indices do not improve event-level BAR prediction when economic and temporal features are already included," which is a much more circumscribed statement.

**3.5. The validation-to-test performance drop undermines the predictive claim.**
The decline from QWK 0.523 (validation) to 0.293 (test) is substantial, representing a 44% degradation. The authors frame this as "informative rather than disqualifying" and attribute it to distributional shift (Iraq War effects). But a model that loses nearly half its performance when applied to the immediate next time period is not a reliable predictive tool, and the paper's title and abstract frame this as a prediction paper. A QWK of 0.293 on the test set, with macro-F1 of 0.315, is only modestly above chance for a 4-class problem. This performance level does not support the operational recommendations made in the Discussion and Conclusion (e.g., "effective prediction requires integration of economic, institutional, and temporal dynamics data"). The authors should either substantially temper the predictive framing or provide evidence that the model performs adequately under the distributional conditions it would actually face in deployment.

---

## 4. Minor Concerns

**4.1.** The reference to the "autoresearch paradigm (Karpathy, 2024)" is unusual in this context. This appears to be a blog post or informal communication rather than a peer-reviewed methodology. If the ablation protocol is inspired by standard practice in machine learning (which it is), cite the relevant ML literature on ablation studies rather than attributing it to a single informal source.

**4.2.** The paper reports QWK as the primary metric but does not discuss its limitations for imbalanced ordinal data. The neutral class contains only 4% of events. QWK can be dominated by the majority class (mild cooperation at 51.9%). Per-class QWK or ordinal-specific metrics (e.g., ranked probability score) would strengthen the evaluation. The macro-F1 scores (0.375-0.383 for the best models) suggest the model struggles with minority classes, but this is not adequately discussed.

**4.3.** The geographic analysis finding that "North America exhibited the highest conflict ratio at 37.7%" is surprising and requires explanation. This likely reflects the coding of US-Mexico and US-Canada water disputes, but in the hydropolitics literature these are typically characterized as managed disagreements within strong institutional frameworks (e.g., the International Boundary and Water Commission, the International Joint Commission). Are these events genuinely comparable on the BAR scale to Jordan basin conflicts? This raises questions about whether the BAR scale captures comparable phenomena across such different geopolitical contexts.

**4.4.** The TFDD dataset ends in 2008. The authors acknowledge this in the limitations but still make strong policy recommendations ("invest in accelerating institutional adaptation," "treat climate data as a background vulnerability indicator"). Given that the post-2008 period has seen significant shifts in transboundary water dynamics (Grand Ethiopian Renaissance Dam, Mekong mainstream dams, accelerating glacier melt in the Indus system, increasing groundwater depletion), the external validity of these recommendations is questionable.

**4.5.** The paper does not discuss the potential circularity in using treaty formation rate to predict BAR outcomes. Treaties in the TFDD are themselves coded as cooperative events. If the treaty formation rate feature captures the same latent process that generates high BAR scores, the relationship is tautological rather than predictive. The authors' data leakage precautions (computing treaty counts strictly prior to each event) address temporal leakage but not this conceptual circularity.

---

## 5. Questions for Authors

**5.1.** The SHAP analysis identifies "number of countries involved" as the top predictor, but this is fundamentally a structural attribute of the event rather than a driver of conflict outcomes. Have the authors tested model performance with this variable removed to assess whether the substantive findings (economic and temporal dominance, climate irrelevance) hold when the model cannot rely on what is effectively a basin-type identifier?

**5.2.** The paper frames the treaty formation rate finding as supporting the "institutional-change hypothesis" of Wolf et al. (2003). However, Wolf et al.'s original argument was about the rate of change in physical or institutional systems relative to institutional capacity to absorb change, not about treaty formation rates per se. How do the authors reconcile their operationalization (treaties per year in a 5-10 year window) with the original theoretical construct, which encompasses a broader set of institutional and physical changes?

**5.3.** Given that 81.3% of conflict events are concentrated in 10 basins, have the authors considered whether their model is effectively a basin classifier rather than a general conflict predictor? What is the model's performance when evaluated exclusively on basins outside the top 10?

---

## 6. Recommendation

**Major Revision.**

The paper addresses a worthwhile question with a reasonably rigorous ML methodology, and the ablation protocol is a useful methodological contribution. However, the theoretical engagement is too shallow for *Global Environmental Change*, the "climate doesn't matter" framing overstates what the evidence supports, the non-independence of observations is inadequately addressed, the validation-to-test performance gap undermines the predictive claims, and the substantive conclusions largely recapitulate existing knowledge. A revised manuscript should: (1) substantially deepen the engagement with the hydropolitics literature beyond Wolf and Zeitoun/Warner; (2) reframe the climate finding as resolution-specific rather than general; (3) implement spatial cross-validation or basin-holdout experiments to address non-independence; (4) provide theoretical justification for the 4-class BAR grouping or test sensitivity to alternative groupings; and (5) more carefully distinguish between confirming existing theory and generating genuinely novel insight.
