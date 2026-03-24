# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Reviewer 9** -- Global Environmental Change

**Expertise**: Environmental security, climate adaptation policy, science-policy interface for water governance

---

## General Assessment

This manuscript presents an ordinal machine learning benchmark on the Transboundary Freshwater Dispute Database, evaluating eight external data sources and applying SHAP-based temporal decomposition to interpret shifting driver importance across geopolitical eras. The ambition is commendable, and the dataset integration effort is substantial. My review focuses specifically on how well the cited literature is woven into the argument, whether theoretical frameworks do genuine analytical work rather than serving as decoration, and whether the policy discussion is grounded in governance scholarship or remains speculative.

Overall, the paper occupies an awkward middle ground. The Introduction assembles an impressive conceptual apparatus from hydropolitics, but the analytical pipeline is largely atheoretical, and the Discussion reconnects to the frameworks only selectively. Several citations appear to serve a legitimation function rather than generating testable expectations. The policy recommendations, while plausible, extrapolate beyond what the correlational evidence supports. These issues are tractable, and the underlying contribution is real, but substantial revision is needed to bring the literature integration up to the standard expected at this journal.

---

## Major Concerns

**1. Theoretical frameworks are cited but not used to generate predictions.**

The Introduction reviews hydro-hegemony (Zeitoun & Warner, 2006), TWINS (Mirumachi & Allan, 2007), counter-hegemonic strategies (Cascao, 2009), institutional resilience (Giordano & Wolf, 2003), and the institutional-change hypothesis (Wolf et al., 2003). This is a well-curated tour of the hydropolitics canon. However, none of these frameworks is used to derive testable hypotheses that the machine learning analysis then evaluates. The ablation protocol tests feature groups defined by data source (climate, governance, economic), not by theoretical construct. For example, the hydro-hegemony framework predicts that power asymmetry between riparians should mediate outcomes, and the paper does include asymmetry features. But this connection is never made explicit before the results: the reader does not encounter a statement such as "if hydro-hegemony theory is correct, then asymmetry features should improve predictive performance beyond what economic indicators alone achieve." The frameworks appear in the Introduction as background and are retrieved in the Discussion as post-hoc interpretive lenses, but they do no predictive work in between. This is a missed opportunity. The paper should articulate at least three to four framework-derived expectations in the Introduction and then evaluate them systematically.

**2. The Discussion does not return symmetrically to all frameworks introduced.**

The Introduction devotes considerable space to the TWINS framework (Mirumachi & Allan, 2007), which argues that conflict and cooperation coexist within the same basin at different intensities. The paper correctly notes that this insight motivates the ordinal (rather than binary) target formulation. But the Discussion never revisits TWINS to ask whether the ordinal model captures this coexistence or whether it remains a limitation. Similarly, Cascao's (2009) counter-hegemonic strategies are cited in the Introduction and mentioned once in the Discussion (Section on economic capacity), but the analysis contains no features that operationalize discursive power, coalition-building, or norm entrepreneurship. If the paper cannot test these constructs, it should say so explicitly in the Introduction rather than raising expectations it does not fulfil. The asymmetry between frameworks introduced and frameworks evaluated creates an impression of selective citation.

**3. The policy discussion extrapolates beyond the evidence base.**

The Discussion section on "institutional velocity" recommends "provisional data-sharing agreements, flexible allocation frameworks, and rapid-response institutional mechanisms" as policy priorities. This is a plausible recommendation, but it is grounded in a SHAP-based association between treaty formation rate and model predictions, not in any evidence about the effectiveness of specific institutional designs. The paper's own epistemological caveat (SHAP values measure marginal feature contributions, not causal effects) should prevent this leap. The citation of Bernauer and Bohmelt (2020) and Conca, Wu & Mei (2006) in this passage is appropriate but insufficient: those studies examine treaty design characteristics and compliance, not the relationship between treaty formation velocity and conflict outcomes. The paper would benefit from either (a) citing governance literature that directly links institutional formation speed to outcome quality, or (b) presenting the policy implications as research questions rather than recommendations. As written, the policy discussion risks reading as speculation dressed in citations.

**4. The epistemological caveat about SHAP is stated but not supported by methodological literature.**

The Discussion opens with an important caveat: "SHAP values measure marginal feature contributions to model predictions, not causal effects on real-world outcomes." This is correct and necessary. However, the sole citation is Lundberg and Lee (2017), which is the original SHAP paper and does not discuss the causal interpretation problem. A growing methodological literature addresses exactly this issue. Janzing, Minorics & Bloebaum (2020, ICML) distinguish between observational and interventional SHAP values. Chen, Harinen, Lee, Yung & Zhao (2020) discuss the gap between feature importance and causal attribution. Kumar, Vaidyanathan, Elenberg & Dahleh (2020) examine problems with SHAP's faithfulness guarantees. The paper should cite at least some of this methodological work to demonstrate awareness that the causal limitation is not merely a general disclaimer but a specific technical property of Shapley-based attribution methods. Without this grounding, the caveat reads as boilerplate rather than informed epistemology.

**5. The climate discussion is thorough but never engages the climate security debate directly.**

The climate section cites Mach et al. (2019), Buhaug (2010), Koubi (2019), Gleditsch (2012), Ide (2015), and Ide et al. (2020). This is a strong set of references. However, the paper does not position its findings within the ongoing methodological debate about whether null climate-conflict results reflect genuine absence of effect or specification choices. Buhaug (2010) and Burke et al. (2009) represent opposing sides of this debate, and the paper cites only Buhaug. The scale-sensitivity argument (climate effects depend on spatial and temporal resolution) is well established but the paper does not cite the key methodological piece by Hsiang, Burke & Miguel (2013), which systematically catalogued how effect sizes vary with resolution. Including the opposing perspective would strengthen the paper's claim that its null finding is resolution-specific rather than substantive, because it would show that this sensitivity is well documented even for studies that do find climate effects.

---

## Minor Concerns

**6. Several citations serve an ornamental rather than argumentative function.**

Warner and Zawahri (2012) is cited in the Introduction with the claim that "power asymmetries shape not only treaty content but compliance patterns, with hegemonic states selectively enforcing provisions that serve their interests." This is a substantive claim, but it is never connected to any feature, result, or discussion point in the paper. The same applies to Zeitoun and Mirumachi (2008), which is cited immediately after Zeitoun and Warner (2006) but adds no distinct analytical purpose. If these works inform the conceptual background, the paper should state what distinct prediction or interpretation each enables. If they do not, they should be removed or moved to a supplementary literature overview.

**7. The Dinar, Katz & Shmueli (2015) citation is underutilized.**

The Introduction cites Dinar et al. (2015) for the finding that "scarcity can itself stimulate institutional innovation." This is a theoretically consequential claim that directly complicates the scarcity-to-conflict narrative. But the paper never returns to this insight. Does the SHAP analysis reveal any pattern consistent with scarcity-driven institutional innovation (e.g., do water-stressed basins show higher treaty formation rates)? If the data cannot speak to this question, the citation raises an expectation it does not fulfil.

**8. The WPS Partnership is cited without a formal reference.**

The Discussion mentions the "Water Peace and Security (WPS) Partnership" and its operational tool, quoting its epistemic disclaimer. But there is no formal reference entry for this source, which appears to be grey literature. For a journal submission, this should either receive a proper citation or be identified clearly as an institutional report with URL and access date.

**9. The framing of "first ordinal-aware benchmark" should be qualified.**

The Abstract and Introduction claim this is "the first ordinal-aware machine learning benchmark on the TFDD." This is a strong priority claim. The paper should verify that no prior work has used ordinal regression or ordinal-aware metrics on the TFDD, even in unpublished or grey literature. If the claim holds, it should be stated more precisely: "first published ordinal-aware benchmark" or similar.

**10. The contribution to ongoing debates is unclear.**

The paper contributes data analysis and methodological cautions (endogeneity of autoregressive features, path-dependence of ablation). These are valuable. But the paper does not clearly state which theoretical debate it advances. Does it support the institutional-change hypothesis over the scarcity hypothesis? It gestures toward this but hedges so heavily (appropriately, given the correlational design) that the reader is left uncertain about the paper's theoretical position. A paragraph in the Discussion explicitly stating "our results are most consistent with [framework X] and less consistent with [framework Y], pending causal confirmation" would sharpen the contribution. As written, the paper risks reading as a technically competent exercise that reports results without taking a theoretical stance.

**11. The connection between Unfried et al. (2022) and the analysis is unclear.**

Unfried et al. (2022) is cited in the Introduction for using GRACE satellite data as instrumental variables to establish causal links between water availability and conflict. This is a methodologically distinct approach (causal identification vs. prediction). The paper should clarify whether it views its own work as complementary to or in tension with the causal identification approach. The Limitations section mentions the need for "causal identification strategies" but does not reference Unfried et al. as an exemplar, which is a missed connection.

**12. The Jiang et al. (2025) citation needs clearer integration.**

Jiang et al. (2025) is cited for a "water dependency framework explaining 80% of historical conflicts through structural vulnerability." This is a strong claim attributed to a very recent paper. The Discussion section on climate does not return to this framework to ask whether water dependency, a structural variable, might mediate the climate-conflict pathway at a resolution the current analysis cannot capture. This would strengthen the argument that the climate null finding is resolution-specific.

---

## Summary Recommendation

**Major revision.** The manuscript presents a methodologically sound and well-documented machine learning analysis, and the self-critical treatment of limitations (autoregressive endogeneity, ablation path-dependence, BAR scale caveats) exceeds the norm for this type of study. However, the literature integration requires substantial improvement. Theoretical frameworks should generate testable expectations, not merely provide post-hoc interpretation. The policy discussion should be more carefully bounded by the associational nature of the evidence. The epistemological caveat about SHAP should be supported by the relevant methodological literature rather than relying on a single citation. And the paper should more clearly articulate which ongoing theoretical debates it advances and how. These revisions would transform a competent empirical report into a genuine contribution to the hydropolitics and environmental security literatures.
