# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Reviewer 13** | *Global Environmental Change*
**Expertise**: Scientific writing, argumentation structure, manuscript organization

---

## General Assessment

This manuscript presents an ordinal machine learning benchmark for predicting transboundary water conflict outcomes using the TFDD. The paper is ambitious in scope and notably self-critical, which is commendable. However, the argumentation suffers from a fundamental tension: the paper oscillates between claiming to have identified "policy-relevant insights" and systematically undermining each of those insights through extensive caveats. The result is a manuscript whose central thesis is difficult to locate. Is this a paper about prediction? About feature importance? About the limitations of prediction? The narrative arc never fully resolves this question, and the Discussion section, while intellectually honest, reads more as an extended limitations section than as a structured interpretation of results.

The paper's strongest argumentative contribution is methodological: the demonstration that autoregressive features inflate validation metrics without improving test generalization, and that sequential ablation is path-dependent. These are clear, well-supported findings. The four "policy insights" in the Conclusion, by contrast, are substantially weaker than their framing suggests. The manuscript would benefit from a clearer separation between what the analysis actually establishes and what it speculates about.

---

## Major Concerns

**1. The paper lacks a clearly articulated thesis.**

A reader finishing the Abstract knows the following: the model achieves moderate performance, autoregressive features are endogenous, treaty formation rate gained importance over time, and spatial generalization is poor. But what is the paper arguing? The three stated contributions in the Introduction (comprehensive feature set, ablation protocol, SHAP temporal decomposition) are methodological descriptions, not arguments. The closest candidate for a thesis appears in the Conclusion: "conflict prediction models trained on historical data do not transfer reliably across geopolitical regimes or geographic contexts." If this is the central argument, it should be foregrounded from the Introduction onward and the entire paper should be organized around demonstrating it. Currently, it arrives as a concluding observation rather than the culmination of a sustained argument.

**2. The Introduction establishes a gap that the paper only partially fills.**

The Introduction identifies three gaps: (a) no prior study accounts for the ordinal structure of the BAR scale, (b) no systematic feature selection has been performed, and (c) no interpretable explanations have been provided. The paper addresses all three, but the payoff from (a) is never demonstrated. The ordinal regression baselines (LogisticAT, OrdinalRidge) are included but substantially underperform XGBoost, which does not explicitly encode ordinal structure. The paper never tests whether ordinal-aware loss functions or ordinal encoding within gradient boosting improves performance over standard multiclass classification. The claim to be the "first ordinal-aware benchmark" therefore rests on using QWK as a metric and grouping BAR into ordered classes, not on demonstrating that ordinal-aware modelling outperforms ordinal-agnostic alternatives. This gap between the promise of the Introduction and the delivery of the Results weakens the narrative arc.

**3. The four policy insights are not adequately supported by the evidence presented.**

This is the most consequential structural problem. Each insight suffers from a specific argumentative weakness:

- *Insight 1 (institutional velocity > institutional stock)*: The 56.5% increase in SHAP importance for treaty formation rate is the strongest piece of evidence. However, the paper itself notes that removing treaty formation rate actually improves test QWK (from 0.132 to 0.161), that treaty formation rate is collinear with cooperation momentum (r = 0.303), and that treaties are themselves coded as cooperative events in the TFDD. These caveats do not merely temper the conclusion; they undermine its evidential basis. A feature whose removal improves out-of-sample prediction cannot be straightforwardly interpreted as policy-relevant, regardless of its SHAP importance.

- *Insight 2 (economic capacity as governance prerequisite)*: The evidence is that economic features are "consistently associated with conflict outcomes across ablation orderings, SHAP rankings, and permutation importance." But the paper provides no test of the specific claim that economic capacity functions as a prerequisite for governance. This is an interpretive leap from correlation to mechanism, acknowledged as such but still presented as a "policy-relevant insight."

- *Insight 3 (climate as structural mediator, not proximate trigger)*: The paper presents a negative result (climate features do not improve event-level prediction at this resolution) and then spends over 600 words interpreting it. The argument that this represents a "scale mismatch" rather than "a refutation of climate's role" is plausible but unfalsifiable within this study. More problematically, climate features were retained under one of three ablation orderings, and grouped permutation importance showed a positive contribution. The paper therefore cannot distinguish between "climate is irrelevant at this resolution" and "climate signal is real but weak and order-dependent," yet frames the finding as though it establishes the structural-mediator interpretation.

- *Insight 4 (governance data gaps as policy blind spot)*: This is the most speculative insight, built on the observation that governance features have high missingness and that alternative imputation strategies improve performance by 0.020-0.022 QWK. The leap from "governance features might contain signal obscured by median imputation" to "data infrastructure investment is itself a conflict prevention tool" is not supported by the analysis.

**4. The Discussion interprets SHAP values as though they reveal mechanisms, despite an explicit caveat against doing so.**

The epistemological caveat at the start of the Discussion is appropriate and well-stated. However, the subsequent four subsections consistently slide from "consistent with" language into mechanistic interpretation. For example, the claim that "investment in institutional velocity... may be more effective than investment in comprehensive but static treaty architectures" is a causal claim about policy effectiveness derived from a feature importance metric. The caveat at the top does not inoculate the text against this pattern; rather, it highlights the disconnect between what the analysis can establish and what the Discussion asserts.

**5. The Results section does not cleanly answer the research questions implied by the Introduction.**

The Introduction implies three research questions corresponding to its three contributions. The Results section, however, is organized around six subsections that do not map cleanly onto these questions. The ablation results (Section 3.1) address contribution 2, the model comparison (Section 3.2) addresses contribution 1 only tangentially (no ordinal-specific comparison), the SHAP analysis (Sections 3.3-3.4) addresses contribution 3, and the geographic and robustness sections (3.5-3.6) address questions not raised in the Introduction. This structural mismatch makes it difficult for the reader to track whether each gap identified in the Introduction has been filled.

---

## Minor Concerns

**6.** The Abstract is overloaded with numerical results (QWK of 0.298, macro-F1 of 0.320, 56.5% increase, 0.062 to 0.417 range) that obscure the narrative. An Abstract for Global Environmental Change should lead with the substantive finding and its policy significance, not with metric values that are meaningless to most readers in this journal's audience.

**7.** The Climate discussion subsection has two headings ("Climate as a structural mediator, not a proximate trigger" and "Climate data specification does not improve event-level prediction at this resolution"). The first appears to be a leftover from an earlier draft. This should be cleaned up, but more importantly, the first heading presupposes a mechanistic interpretation that the analysis cannot support; the second heading is more accurate.

**8.** The Introduction's literature review is thorough but front-loaded. The paragraph on hydro-hegemony, TWINS, counter-hegemonic strategies, institutional resilience, and treaty design (paragraph 2) introduces concepts that are not all operationalized in the analysis. Zeitoun and Mirumachi (2008), Cascao (2009), Conca, Wu and Mei (2006), and Warner and Zawahri (2012) are cited but their frameworks are not tested. This creates an expectation the paper does not fulfill. Either these citations should be moved to the Discussion (where they are used for interpretation) or the paper should explain why they are included in the Introduction despite not being operationalized.

**9.** The Conclusion presents the four policy insights as "associational hypotheses requiring causal confirmation," which is appropriately hedged. However, the framing as "policy-relevant insights" in both the Discussion and Conclusion is in tension with this hedge. If these are hypotheses requiring causal confirmation, they are not yet policy insights; they are research directions. The language should be consistent.

**10.** The validation-to-test gap is described as "26%" for the primary model (test QWK 0.298 vs validation QWK 0.403). This percentage is computed as (0.403 - 0.298) / 0.403, but this calculation is never shown. For a paper that emphasizes transparency, this metric should be explicitly defined rather than stated as though self-evident.

**11.** The Limitations section (10 items) is exhaustive to the point of undermining confidence in the findings. While intellectual honesty is valued, listing ten limitations without prioritizing them by severity makes it difficult for the reader to distinguish fundamental constraints from minor caveats. Grouping these into two or three thematic clusters (data limitations, methodological limitations, generalizability limitations) would improve readability.

**12.** The paper references "Extended Data" figures and tables throughout, but several key claims depend on these supplementary materials (e.g., bootstrap CIs for temporal SHAP decomposition, sensitivity analyses with 3-class and 5-class groupings). The main text should contain sufficient evidence for its primary claims without requiring the reader to consult supplementary materials for critical evidence.

**13.** The spatial cross-validation analysis (Table 5) is presented in the Results but not connected to a research question in the Introduction. Geographic transferability is an important finding, but it appears as a post-hoc analysis rather than a pre-specified evaluation. If the authors consider this a core contribution, it should be motivated in the Introduction.

**14.** The sentence "Freshwater scarcity is intensifying as a geopolitical pressure point" opens the Introduction but is not cited. For a claim this broad, a citation is expected.

---

## Summary Recommendation

The manuscript contains technically sound work with commendable transparency about limitations. However, the argumentation structure requires substantial revision before it meets the standards of Global Environmental Change. The central thesis must be identified and foregrounded. The four policy insights must either be supported more rigorously or reframed as exploratory hypotheses without the "policy insight" label. The Discussion must maintain the epistemic discipline established in its opening caveat throughout all four subsections. The Results section should be reorganized to map onto the research questions established in the Introduction.

I recommend **major revision** with attention to the structural and argumentative issues raised above. The technical analysis does not need to change; the framing, narrative arc, and inferential claims do.
