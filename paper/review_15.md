# Peer Review 15: Global Environmental Change Editorial Board Assessment

**Reviewer**: Peer Reviewer 15, Editorial Board Member, Global Environmental Change
**Manuscript**: "Predicting Transboundary Water Conflict Outcomes: An Ordinal Machine Learning Benchmark on the TFDD"
**Date**: 24 March 2026
**Review type**: Scope fit and editorial reception assessment

---

## Overall Assessment

This manuscript presents an ordinal machine learning benchmark for predicting transboundary water conflict outcomes using the TFDD, integrating eight external data sources and applying SHAP-based interpretability analysis. The work is technically competent, methodologically self-aware, and engages seriously with the hydropolitics literature. However, as I detail below, the manuscript falls into an uncomfortable gap between a machine learning methods contribution and a social science contribution, and in its current form does not meet the threshold for Global Environmental Change (GEC). My recommendation is **redirect with encouragement to reframe and resubmit**.

---

## 1. Does It Meet GEC's Requirement for a "Significant Social Science Component"?

GEC requires that manuscripts make a substantive contribution to understanding human-environment interactions through social science theory, not merely apply computational tools to environmental data. This is where the manuscript struggles most.

The paper engages extensively with the hydropolitics literature. The introduction competently reviews Wolf, Zeitoun, Mirumachi, Cascao, and other key theorists. The discussion threads SHAP findings back to the institutional-change hypothesis and the hydro-hegemony framework. These are genuine strengths. However, the engagement remains interpretive rather than generative. The manuscript does not advance, test, or refine any social science theory. Instead, it uses established frameworks as post-hoc lenses for interpreting feature importance rankings. The institutional velocity finding (treaty formation rate increasing 56.5% in SHAP importance from the Cold War to the post-2000 era) is positioned as "consistent with" Wolf et al.'s institutional-change hypothesis, but the analysis cannot distinguish this interpretation from confounding by secular trends in treaty-making norms, geopolitical restructuring, or data coverage shifts. The authors acknowledge this repeatedly, which is commendable, but it means the social science contribution reduces to: "our correlational ML results do not contradict existing theory."

GEC has published ML-informed papers on environmental conflict (Ide et al., 2020, cited in the manuscript, appeared in GEC), but those papers typically start from a theoretical proposition, operationalize it as a testable hypothesis, and use quantitative methods to evaluate it. This manuscript starts from a prediction task and retrofits theoretical interpretation. The direction of inference matters for GEC.

The discussion section contains four "policy-relevant insights," but each is hedged as an "associational hypothesis requiring causal confirmation." This level of epistemic caution is scientifically appropriate but leaves GEC reviewers asking: what do we now know about human-environment dynamics that we did not know before? The answer, candidly, is not enough for GEC's standards.

---

## 2. Is the ML Positioned as a Tool for Governance Insight Rather Than the Primary Contribution?

No. Despite the authors' efforts to foreground policy implications, the manuscript's primary contribution is methodological: the first ordinal-aware ML benchmark on the TFDD, a systematic ablation protocol, the autoregressive endogeneity demonstration, and the SHAP temporal decomposition technique. The abstract leads with the benchmark, the methods section occupies roughly 40% of the paper, and the most technically novel findings (path-dependence of sequential ablation, autoregressive inflation of validation metrics) are ML methodology contributions with limited governance content.

The title itself signals this: "An Ordinal Machine Learning Benchmark on the TFDD" is a methods paper title, not a GEC title. A GEC-appropriate framing would center the governance question (e.g., "Why institutional velocity, not institutional stock, predicts transboundary water cooperation") and relegate the ML pipeline to a methods section. In its current form, the ML *is* the contribution, and the governance insights are secondary observations.

This is not a criticism of the work itself. It is a scope observation. The ablation path-dependence finding, the autoregressive endogeneity demonstration, and the spatial CV geographic variation analysis are genuinely useful for the conflict prediction community. They are simply not what GEC publishes.

---

## 3. Are the Policy Implications Substantive Enough for GEC?

The four policy messages are: (1) invest in institutional velocity over static treaty architectures; (2) economic capacity-building serves dual water governance functions; (3) climate operates as a structural mediator, not a proximate trigger at event scale; (4) governance data gaps constrain conflict prediction in the regions that need it most.

These are reasonable inferences, but none is new. Wolf et al. (2003) articulated the institutional-change hypothesis two decades ago. The economic capacity argument restates Zeitoun and Warner (2006). The climate finding aligns with Mach et al. (2019) and the broader consensus documented by Buhaug (2010) and Koubi (2019). The governance data gap observation is well known in the development indicators community. What the manuscript adds is ML-derived quantitative decoration of existing qualitative insights, which is not nothing, but it does not reach the novelty threshold GEC requires for policy implications.

More fundamentally, the policy implications are generic. GEC editors will ask: what should a water diplomat or basin commission do differently based on this paper? The answer, as written, is "invest in flexible institutional mechanisms and governance data collection," which is advice already embedded in UNDP and World Bank water governance frameworks. The manuscript would need to derive specific, actionable, and non-obvious policy recommendations to satisfy GEC's policy relevance standard.

---

## 4. Would This Survive the Desk Review (2-2.5 Week Triage)?

I do not think it would in its current form. GEC desk rejection rates run 40-60%, and the handling editors screen aggressively for three things: (a) a clear social science research question, (b) a contribution that advances understanding of human-environment interactions, and (c) policy relevance that goes beyond generic recommendations. The manuscript is weakest on (a) and (c).

The research question as stated is predictive ("can we forecast conflict intensity on the BAR scale?"), not explanatory ("what institutional or structural conditions drive the transition from cooperation to conflict?"). GEC publishes the latter. A predictive framing signals to the handling editor that this belongs in a methods-oriented outlet.

The modest predictive performance (test QWK of 0.298, macro-F1 of 0.320) will also raise editorial concern. While the authors are transparent about this and frame it as characterizing the difficulty of the task, a GEC editor may reasonably conclude that a model explaining less than 30% of ordinal agreement does not generate reliable enough patterns to inform governance.

---

## 5. Comparison to Recent GEC Publications on Water Conflict and Environmental Security

GEC's recent publications in this space are instructive. Ide et al. (2020) in GEC used multi-method evidence to demonstrate *when and how* climate disasters contribute to armed conflict risk, with a clear causal mechanism and governance-conditional framing. Bernauer and Bohmelt (2020), also in GEC, provided econometric evidence on institutional design characteristics and treaty effectiveness, with instrumental variable approaches supporting causal claims. These papers share a common structure: they start from a social science puzzle, use quantitative methods to adjudicate between competing theoretical explanations, and derive policy implications that follow logically from the causal findings.

This manuscript does not follow that structure. It starts from a prediction task, applies an ML pipeline, and interprets the outputs through existing theory. The interpretive quality is high, the epistemic caution is exemplary, but the inferential architecture is fundamentally different from what GEC publishes. The closest recent GEC analogue would be Ge et al. (2022) in Nature Communications (not GEC), which used boosted regression trees for conflict prediction but centered the climate sensitivity finding rather than the ML benchmark.

---

## 6. Recommendation

**Redirect.** The manuscript should not be submitted to GEC in its current form. However, the underlying work has genuine merit, and I see two viable paths forward.

### Path A: Reframe for GEC (substantial revision required)

If the authors wish to target GEC, the manuscript would need to be fundamentally restructured:

- Lead with a social science research question (e.g., "Does institutional velocity predict cooperative outcomes independently of institutional stock, and has this relationship strengthened in the post-Cold War era?")
- Reduce the ML methods to a supporting role, moving ablation details and model comparison to supplementary materials
- Develop the institutional velocity finding into a standalone theoretical contribution, ideally with causal identification (even a basic IV or regression discontinuity approach using exogenous shocks to treaty-making capacity would strengthen the causal claim)
- Derive specific, basin-level policy recommendations rather than generic governance advice
- Engage more deeply with the TWINS framework to address the limitation that a single BAR score per event collapses a multi-dimensional interaction space

This would be a different paper. Whether the data and analysis support it is an open question.

### Path B: Submit to a more appropriate venue (recommended)

The manuscript as written is a strong fit for several alternative journals:

1. **Environmental Modelling & Software** -- The ablation protocol, autoregressive endogeneity demonstration, and spatial CV analysis are methodological contributions that this journal publishes and values. The TFDD benchmark would serve the modelling community well.

2. **Journal of Peace Research** -- If the authors can strengthen the conflict prediction angle and engage more with the quantitative peace research literature (Hegre, Buhaug, von Uexkull), this journal publishes ML applications to conflict forecasting.

3. **Water Resources Research** -- The hydrological and institutional integration, particularly the climate resolution finding and the governance data gap analysis, aligns with WRR's scope for transboundary water governance.

4. **PLOS ONE or Scientific Reports** -- If the authors wish to publish the benchmark quickly without major restructuring, these journals accept technically sound ML benchmarks with adequate domain context.

5. **International Environmental Agreements: Politics, Law and Economics** -- The institutional velocity finding and treaty analysis would fit well here with modest reframing toward the treaty effectiveness literature.

My strongest recommendation would be **Environmental Modelling & Software** or **Journal of Peace Research**, depending on whether the authors wish to emphasize the methodological or the conflict prediction contribution.

---

## Summary of Strengths and Weaknesses

**Strengths:**
- Technically rigorous ML pipeline with exemplary transparency about limitations
- Serious engagement with the hydropolitics literature, well beyond what most ML papers achieve
- The autoregressive endogeneity finding is a genuine service to the conflict prediction community
- Epistemic caution throughout the discussion is refreshing and scientifically appropriate
- Comprehensive robustness testing (ablation path-dependence, spatial CV, sensitivity analyses)

**Weaknesses for GEC specifically:**
- No social science research question; the framing is predictive, not explanatory
- ML benchmark is the primary contribution; governance insights are secondary interpretations
- Policy implications restate existing knowledge with quantitative support but no new actionable specificity
- Modest predictive performance limits the reliability of pattern extraction for governance purposes
- No causal identification strategy, which the authors acknowledge but which GEC increasingly expects

---

## Final Verdict

This is a competent and honest piece of applied ML research with genuine methodological contributions and thoughtful domain engagement. It is not, however, a GEC paper. The social science component is interpretive rather than generative, the ML pipeline dominates the contribution structure, and the policy implications do not advance beyond existing consensus. I recommend the authors target Environmental Modelling & Software or Journal of Peace Research, where the methodological rigour and domain-aware framing would be well received and appropriately valued.
