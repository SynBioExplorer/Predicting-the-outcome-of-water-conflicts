# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Journal**: *Global Environmental Change*

**Reviewer**: Reviewer 6

**Expertise**: Climate-conflict linkages, environmental security, climate adaptation, quantitative environmental social science

**Date**: 23 March 2026

---

## 1. Summary

This paper develops an ordinal machine learning benchmark for predicting the intensity of transboundary water conflict events using the TFDD, evaluating the incremental predictive value of eight external data sources through a sequential ablation framework. The central empirical claim is that basin-averaged annual climate indices do not improve event-level prediction, while economic indicators and temporal dynamics (particularly the rate of treaty formation) emerge as candidate predictors. The best-performing model (XGBoost, Optuna-tuned) achieves a validation QWK of 0.502 that degrades substantially to 0.290 on the held-out test set, and SHAP temporal decomposition reveals shifting driver importance across geopolitical eras.

## 2. Overall Assessment and Recommendation

**Recommendation: Major Revision**

The paper tackles an important question at the intersection of hydropolitics and quantitative conflict forecasting, and the ordinal-aware framing represents a genuine methodological advance over binary treatments of the BAR scale. The ablation design with robustness testing across orderings, the candid treatment of the validation-to-test gap, and the SHAP temporal decomposition are commendable. However, the climate finding, which will inevitably become the most cited result from this paper, suffers from serious interpretive issues that risk contributing to a misleading narrative in the climate-security literature. The paper requires substantial revision in how it frames the climate null result, deeper engagement with the theoretical mechanisms linking climate to conflict, and additional analyses to distinguish between "climate does not matter" and "this particular climate data specification is inadequate." I detail these concerns below.

## 3. Major Concerns

**1. The climate null result is an artefact of measurement specification, not a substantive finding about climate-conflict linkages, and the paper does not adequately distinguish between these two interpretations.**

The authors test five climate features: basin-averaged annual precipitation, potential evapotranspiration, SPEI-3, and anomalies thereof, all derived from CRU TS 4.09 at 0.5-degree resolution. These are then averaged across entire basin polygons and matched to individual diplomatic events. This specification choice stacks multiple layers of spatial and temporal aggregation: 0.5-degree grid cells are area-weighted across basins that range from hundreds to millions of square kilometres, monthly data are collapsed to annual means, and the resulting single number is assigned to events that may be triggered by sub-seasonal, sub-basin hydrological shocks.

The authors acknowledge the scale mismatch in the Discussion, but the acknowledgment reads as a caveat rather than as the central interpretation. The paper's structure, from the abstract ("basin-averaged annual climate indicators do not improve event-level prediction") through the Discussion heading ("Basin-averaged annual climate indicators do not improve event-level prediction at this resolution"), presents the negative result as a finding about climate's role, qualified by resolution. In the climate-security literature, such results are routinely cited without the qualification. Mach et al. (2019) explicitly warned against this: their expert elicitation found that evidence for climate-conflict linkages was strongest at sub-national scales and for specific conflict types, and that null results at coarse aggregation should not be interpreted as evidence of absence.

I request that the authors:
(a) Move the scale-mismatch discussion from a subsection of the Discussion to a prominent position in the Results, immediately following the ablation table, so that readers encounter the interpretive frame before absorbing the headline number.
(b) Test at least one finer-grained climate specification. SPEI-3 is available monthly and at the basin level. Computing, for example, the minimum 3-month SPEI in the 12 months preceding each event, or the maximum negative precipitation anomaly in the upstream portion of the basin, would provide a fairer test. If these also fail, the negative result becomes more convincing. If they succeed, it confirms the scale-mismatch interpretation.
(c) Reframe the abstract and conclusion to state explicitly that the finding is about a specific data specification, not about climate as a driver category.

**2. The paper's engagement with the climate-security literature is selective and omits key theoretical frameworks that would contextualise the finding.**

The Discussion cites Mach et al. (2019), Gleditsch (2012), and Ide (2015), but the engagement is superficial. Several critical gaps:

First, Ide et al. (2020) demonstrated that climate effects on conflict are moderated by governance quality, meaning that climate's predictive contribution should be tested not as a main effect but as an interaction with institutional and governance variables. The authors test climate as an additive feature group. If climate operates through governance (as Ide et al. argue), then adding climate features after governance features are already in the model will show no marginal gain, because the variance is already captured. The ablation design is structurally incapable of detecting interaction effects, and the paper should acknowledge this limitation explicitly.

Second, Mach et al. (2019) distinguished between climate as a "threat multiplier" (operating through existing vulnerabilities) and climate as a direct driver. The threat-multiplier framing predicts exactly the result the authors observe: climate alone should not predict event outcomes because its effects are mediated through socioeconomic and institutional channels. But this is not a null result for climate; it is a confirmation of the indirect-pathway hypothesis. The paper should engage with this distinction rather than presenting the finding as evidence against climate's relevance.

Third, Buhaug (2010) and subsequent debate with Burke et al. (2009) over climate-conflict linkages in Africa established that the statistical relationship between climate and conflict is highly sensitive to model specification, variable definition, and spatial scale. The authors' result is entirely consistent with this specification sensitivity, yet Buhaug is not cited.

Fourth, Koubi (2019, *Annual Review of Political Science*) provides a comprehensive review of the climate-conflict evidence base that would help position this finding. The review documents how results vary dramatically with measurement choices, exactly the pattern observed here.

I request engagement with these works and, critically, a test of climate-governance interaction effects (e.g., including climate x governance interaction terms or testing climate's ablation contribution conditional on governance being present vs absent).

**3. The "background stressor" framing is theoretically imprecise and risks being misinterpreted.**

The paper repeatedly characterises climate as a "background stressor rather than a proximate trigger." While this framing has intuitive appeal, it conflates two distinct theoretical claims:

(a) Climate operates on longer timescales than the event resolution of the TFDD, so its signal is temporally diffuse at event level. This is a measurement/resolution claim.

(b) Climate operates as a structural condition that shapes vulnerability but does not directly trigger specific events. This is a causal-mechanism claim.

These are very different arguments. Claim (a) could be resolved with better data. Claim (b) is a substantive theoretical position about causal pathways. The paper oscillates between the two without clearly distinguishing them.

Allan (2001) and Zeitoun's work on "virtual water" demonstrated that water-scarce states adapt through trade, technology, and institutional innovation, meaning that climate stress is absorbed by economic and institutional buffers. This supports claim (b). But Unfried et al. (2022), whom the authors cite in the Introduction but do not engage with in the Discussion, used GRACE satellite data to establish causal links between water availability and localised conflict, supporting the direct-pathway hypothesis at finer spatial scales.

The paper should clearly separate these two arguments and indicate which one the evidence supports. The ablation design can speak to claim (a) but not to claim (b), because the latter requires causal identification strategies (instrumental variables, natural experiments) that are beyond the scope of this correlational exercise.

**4. The robustness of the climate null result is undermined by the authors' own Table 2.**

Table 2 shows that under the reversed ablation ordering, climate variables are retained (delta > 0.005). This means the climate null result is not robust to ordering. The paper acknowledges this but continues to present the negative climate finding as a headline result. If climate is retained under one of three orderings, the honest summary is "climate's incremental value is ambiguous and order-dependent," not "climate does not improve prediction." The reversed ordering retains baseline + temporal + economic + climate with a validation QWK of 0.400, compared to 0.404 for the primary ordering that excludes climate. The difference is 0.004, well within any reasonable confidence interval.

I request that the authors:
(a) Report bootstrap confidence intervals on the QWK difference between the best climate-included and climate-excluded configurations.
(b) If these intervals overlap (as I expect they will), revise the framing to "climate's incremental contribution is not reliably distinguishable from zero at this resolution" rather than "climate does not improve prediction."

**5. The permutation importance analysis does not adequately address climate.**

The authors report that no climate variable appears in the top 15 features by SHAP importance and present this as converging evidence for the climate null result. However, this is a different question from whether climate improves aggregate model performance. Individual climate features may have small but collectively meaningful contributions that do not register in a per-feature importance ranking. Grouped permutation importance (permuting all climate features simultaneously) would provide a fairer test. Was this conducted? If so, the result should be reported. If not, it should be.

**6. The temporal SHAP decomposition, while innovative, has methodological weaknesses that undermine the treaty formation rate finding.**

The 56.5% increase in treaty formation rate importance from the Cold War to the post-2000 era is presented as support for the institutional-change hypothesis. However:

(a) The three eras have very different sample sizes and basin compositions. The post-2000 era (2000-2008) covers only 8 years, while the Cold War era covers approximately 42 years. SHAP values computed over these unequal samples are not directly comparable without normalisation for sample size and composition.

(b) Treaty formation rate is itself a function of time. Basins that were treaty-poor in the Cold War era have more room for treaty formation in later periods. The increasing importance of treaty formation rate may reflect a mechanical relationship: as the baseline of treaties rises, marginal treaty formation becomes a stronger signal simply because the variance in formation rate increases.

(c) The authors do not test whether the SHAP importance shift is statistically significant. A bootstrap procedure that resamples within each era and computes the distribution of importance changes would establish whether 56.5% is distinguishable from random fluctuation.

## 4. Minor Concerns

1. **Citation of Gleditsch (2012) is incomplete.** The paper cites Gleditsch in passing but does not engage with his central argument: that the climate-conflict literature suffers from publication bias toward positive findings, and that null results at specific scales should be expected given the indirect nature of the causal pathway. This argument actually supports the authors' finding, and engaging with it would strengthen the Discussion.

2. **The SPEI-3 choice is not justified.** SPEI can be computed at multiple timescales (1, 3, 6, 12, 24 months), each capturing different drought dynamics. Why was SPEI-3 selected? Agricultural drought, which most directly affects food security and livelihood stress, is typically better captured by SPEI-6 or SPEI-12. Hydrological drought affecting reservoir operations may require SPEI-24. Testing a single timescale and concluding that "drought indices do not predict conflict" understates the dimensionality of the climate signal.

3. **The paper does not discuss seasonal timing of events.** Many transboundary water disputes are triggered during low-flow seasons. If event dates are available at monthly resolution, testing whether climate conditions in the 3-6 months immediately preceding the event (rather than annual averages) predict outcomes would be informative.

4. **The framing of North America's conflict ratio deserves more nuance.** The paper notes that North America has the highest continental conflict ratio (37.7%) and attributes this to US-Mexico and US-Canada disputes. This is a surprising finding that could reflect coding artefacts in the BAR scale rather than genuine conflict intensity. The paper acknowledges that "institutionalised disagreements within robust treaty regimes may receive similar codes to disputes between states lacking formal cooperative arrangements," but this deserves more discussion. If the BAR scale does not distinguish between the severity of a US-Canada Columbia Treaty renegotiation and an India-Pakistan Indus dispute, this is a fundamental measurement problem that affects the entire analysis.

5. **The governance missingness issue (60-64%) is more serious than acknowledged.** Median imputation of features with >60% missingness is likely to render these features uninformative by compressing their variance. The finding that governance does not improve prediction may therefore be an artefact of imputation rather than a genuine null result. The authors note in the Limitations that alternative imputation strategies improve test QWK, but this should be explored more systematically. Multiple imputation or model-based imputation (e.g., MICE) would be more appropriate for this level of missingness.

6. **The comparison with the WPS Partnership tool is incomplete.** The authors note that the WPS tool has an 86% capture rate with a 50% false positive rate, but do not translate their own model's performance into comparable terms (sensitivity/specificity for the conflict class). Providing this comparison would help practitioners assess the relative utility of the two approaches.

7. **Minor writing issue.** The sentence "This represents a scale mismatch between the climate data specification and the level of analysis, not a refutation of climate's role in water conflict" is buried in paragraph two of the climate discussion. This should be the lead sentence of the section, not a subordinate clause.

## 5. Specific Questions for the Authors

1. Have you tested any sub-annual or sub-basin climate specifications? If not, would you be willing to test at least one (e.g., minimum monthly SPEI in the 12 months preceding each event, or upstream-only precipitation anomaly) to determine whether the null result persists at finer resolution?

2. What is the correlation between your climate features and the economic features (particularly GDP per capita)? If climate stress depresses GDP, and GDP is already in the model, climate's incremental contribution will be zero even if it is causally relevant. Have you examined this collinearity?

3. The reversed ablation ordering retains climate. What specific climate features contribute to that ordering's QWK, and do they capture different information than the features discarded under the primary ordering?

4. Have you considered testing climate as a moderator rather than a direct predictor? For instance, does the predictive value of governance or economic features change in basins experiencing drought versus non-drought conditions? This would test the threat-multiplier hypothesis directly.

5. The TFDD ends at 2008. The period 2008-2025 has seen significant climate-driven water stress events (Cape Town Day Zero, Central Asian glacier retreat, Mesopotamian marsh desiccation). Do you have plans to extend the analysis to more recent data, which might capture climate-conflict linkages that were weaker in the 1948-2008 period?

6. What fraction of events in the TFDD have sub-annual date precision? If most events are coded to the exact month, monthly climate matching would be feasible and would substantially reduce the temporal aggregation problem.

7. The institutional-change hypothesis (treaty formation rate > treaty stock) is compelling, but could you clarify the direction of causality? Basins entering a cooperative phase will simultaneously generate more treaties and more cooperative BAR events. How do you distinguish the predictive value of treaty formation rate from this shared trajectory?

8. Regarding the SHAP temporal decomposition: have you tested whether the importance shifts persist when you control for changing basin composition across eras (e.g., by restricting to basins active in all three periods)?

---

## Summary Judgment

This paper makes a valuable contribution to the quantitative hydropolitics literature through its ordinal-aware framing, systematic ablation design, and temporal SHAP decomposition. The honesty about the validation-to-test gap and the autoregressive endogeneity concern is commendable and sets a positive standard for the field. However, the climate finding, which will be the most impactful and most cited result, is inadequately supported by the current analysis. The null result is not robust to ablation ordering, is confounded by scale mismatch and collinearity, and is presented with framing that will be misinterpreted in the broader climate-security debate. The paper should test at least one finer-grained climate specification, engage more deeply with the theoretical literature on climate-conflict pathways, and reframe the finding as resolution-specific rather than substantive. With these revisions, the paper would make a strong contribution to Global Environmental Change.
