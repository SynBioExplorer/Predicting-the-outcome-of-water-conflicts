# Peer Review 8 -- Global Environmental Change

## Manuscript: "Predicting Transboundary Water Conflict Outcomes: An Ordinal Machine Learning Benchmark on the TFDD"

**Reviewer expertise**: Contemporary water conflict scholarship (2020-2026), emerging datasets, quantitative conflict prediction

**Review focus**: Currency and completeness of the literature

---

## General Assessment

This manuscript presents a well-executed ordinal ML benchmark on the Transboundary Freshwater Dispute Database. The methodological contribution is genuine, and the ablation protocol with robustness testing across orderings is a commendable design choice. However, the literature engagement is substantially outdated and incomplete. The manuscript cites foundational hydropolitics scholarship from 2003-2012 competently, but its awareness of the field as it exists in 2024-2026 is poor. Several major developments in transboundary water conflict datasets, ML-based conflict prediction, and the institutional landscape surrounding the TFDD itself appear to have been missed entirely. For a journal of Global Environmental Change's standing, the literature must reflect the current state of the art, not the state of the art circa 2020. Below I enumerate specific concerns, organized by severity.

---

## Major Concerns

**1. The TFDD has been rebranded and updated; the manuscript does not reflect this.**

The manuscript refers throughout to the "Transboundary Freshwater Dispute Database (TFDD), maintained by Oregon State University" and notes that data coverage ends at 2008. As of 2022-2023, Oregon State University's Program in Water Conflict Management and Transformation undertook a significant update and rebranding of the database infrastructure. The International Water Events Database now extends event coverage beyond the 2008 cutoff used here. The authors should acknowledge whether they attempted to access more recent releases, explain why 2008 remains the temporal boundary, and update institutional references accordingly. Referring to the database by its legacy name without acknowledging updates gives the impression that the authors are unaware of developments in their primary data source. The Data and Code Availability section URL (transboundarywaters.ceoas.oregonstate.edu) should be verified against current hosting.

**2. The Water Peace and Security Partnership tool is inadequately characterized.**

The manuscript's single-sentence description of the WPS Partnership tool ("a random forest and LSTM pipeline achieving 86% capture rate for conflict events but with a 50% false positive rate") is both outdated and incomplete. The WPS tool has undergone significant methodological evolution since its initial deployment. By 2023-2024, WPS had moved to an ensemble approach incorporating gradient-boosted models alongside the LSTM component, expanded geographic coverage, and refined its operational definition of "water-related conflict" to include subnational events. The tool now provides 12-month rolling forecasts at the sub-national administrative level, a fundamentally different spatial resolution than the basin-level events in the TFDD. The manuscript quotes the WPS epistemic disclaimer ("is not intended to elucidate causal relationships between the predictor variables and conflict") approvingly, which is appropriate, but does not discuss how the WPS approach differs structurally from the present study: WPS predicts conflict onset (binary), whereas this manuscript predicts conflict intensity (ordinal). This distinction deserves explicit treatment because it defines the manuscript's niche contribution. The 86%/50% figures should be sourced and dated, as WPS performance statistics have been revised in subsequent publications and technical reports.

**3. Competing and complementary datasets are not discussed.**

The manuscript treats the TFDD as if it were the only structured dataset for transboundary water interactions. This is a significant omission. At minimum, the following should be discussed and positioned against:

- The **International River Cooperation and Conflict (IRCC)** dataset, which provides event-level coding of transboundary water interactions with a different coding schema than the BAR scale and has been used in several recent publications. The IRCC covers a partially overlapping but distinct event universe, and any benchmark paper should acknowledge it.
- The **Issue Correlates of War (ICOW)** river claims dataset, which codes territorial and resource claims over international rivers using a different unit of analysis (the claim-dyad-year) than the TFDD's event-level coding. ICOW data have been used extensively in the quantitative international relations literature for studying river-related militarized disputes.
- The **Pacific Institute Water Conflict Chronology**, maintained by Peter Gleick's group, which provides a continuously updated timeline of water-related conflicts including subnational and non-state events excluded from the TFDD. The Chronology has been substantially expanded in 2023-2025 and represents the most current publicly available record of water conflict events.
- The **Global Water Security & Sanitation Partnership** datasets maintained by the World Bank, which provide structured indicators on water governance that could serve as alternative feature sources.

A benchmark paper that does not position its data source against alternatives cannot claim to establish a definitive baseline for the field.

**4. Recent ML-for-conflict-prediction literature (2022-2026) is largely absent.**

The manuscript cites Ge et al. (2022) and Jiang et al. (2025) as the primary recent quantitative references, but misses a substantial body of work on ML-based conflict prediction that has appeared since 2022. Key omissions include:

- Work on transformer-based and graph neural network architectures for conflict forecasting. The conflict prediction community has moved substantially beyond random forests and gradient-boosted trees, with several groups demonstrating that attention-based models capture temporal dependencies in event sequences more effectively than tree ensembles. The manuscript should position its XGBoost approach against these developments, even if the TFDD sample size does not support such architectures.
- The ViEWS (Violence Early-Warning System) project at Uppsala University, which represents the current methodological frontier for sub-national conflict prediction. While ViEWS focuses on armed conflict rather than water-specific events, its validation framework (true out-of-sample temporal forecasting, calibration assessment, ensemble combination) sets the standard against which any conflict prediction benchmark should be measured. The manuscript's temporal split validation is conceptually aligned with ViEWS principles but does not cite or discuss this connection.
- Mueller & Rauh (2022, American Political Science Review) on using text-as-data methods (NLP on news corpora) for conflict prediction, which represents a methodological frontier that the manuscript does not acknowledge.
- Hegre et al.'s continued work on the PRIO conflict prediction competition, which has established community benchmarks and evaluation protocols that the present study could reference.

The absence of these references makes it difficult to assess whether the manuscript's XGBoost benchmark is state-of-the-art or already superseded by developments in adjacent fields.

**5. No engagement with the causal inference revolution in conflict studies.**

The manuscript commendably includes caveats about associational versus causal claims, and cites Unfried et al. (2022) on instrumental variables. However, it does not engage with the broader shift toward causal identification in the climate-conflict literature that has occurred since 2020. Burke, Hsiang & Miguel's framework for causal climate-conflict analysis, the growing use of synthetic control methods and regression discontinuity designs in water conflict settings, and Koubi et al.'s (2022) updated review of causal pathways between environmental change and conflict are all absent. For a paper that repeatedly discusses whether climate "causes" conflict versus serving as a structural mediator, this omission is consequential. The manuscript's discussion of climate as a structural mediator (Section 4.3) would benefit substantially from engaging with Ide et al.'s (2023) updated typology of climate-conflict causal pathways, which distinguishes direct, indirect, and moderated mechanisms more precisely than the 2020 paper currently cited.

---

## Minor Concerns

**6. The Zeitoun and Warner (2006) hydro-hegemony framework has been substantially updated.**

The manuscript relies on the original 2006 formulation. Zeitoun, Cascao, and colleagues have published updated frameworks (the Framework of Hydro-Hegemony, or FHH, revisions in 2020-2021) that refine the original power typology and incorporate soft power and discursive dimensions more explicitly. Since the manuscript claims that its features "capture only material power asymmetry" and acknowledges missing discursive power, citing the updated framework would strengthen this self-critique.

**7. The Mirumachi TWINS framework has evolved.**

The 2007 conference paper cited for the TWINS framework predates the framework's more rigorous formulation in Mirumachi (2015, Routledge) and subsequent updates. The co-existence of conflict and cooperation is now a well-established empirical finding with more recent quantitative support than the 2007 conference proceedings suggest.

**8. Missing references on dam-related conflict acceleration.**

The manuscript notes "new dam construction (particularly in the Mekong and Nile basins)" as a post-2008 development but does not cite the substantial recent literature on how dam construction has reshaped transboundary water politics. Siciliano et al. (2019), Siciliano & Urban (2024), and Kattelus et al.'s work on the Grand Ethiopian Renaissance Dam (GERD) dispute are directly relevant. The GERD case is arguably the single most significant transboundary water conflict development since 2011 and receives no mention.

**9. No reference to the UN Water Convention developments.**

The entry into force of the UN Watercourses Convention (2014) and the expansion of the UNECE Water Convention to non-UNECE states (opened 2016, with significant accessions 2022-2025) represent major institutional developments in the study period's aftermath. These are directly relevant to the "institutional velocity" finding and should be acknowledged in the Discussion.

**10. The reference to "conflict events have accelerated since 2017" (Introduction, line 21) is unsourced.**

This is a strong empirical claim that requires a citation. If this refers to Pacific Institute data, the World Resources Institute's Aqueduct analysis, or another source, it should be attributed. If it refers to events outside the TFDD's temporal coverage, the evidentiary basis should be made explicit.

**11. Polity V has been superseded.**

The manuscript uses Polity V democracy scores as governance features. The Polity project underwent a controversial transition (Polity IV to V) and has faced criticism regarding coding reliability and conceptual validity. V-Dem (Varieties of Democracy) has become the preferred governance dataset in the quantitative conflict literature since approximately 2020. The manuscript should acknowledge this and discuss whether V-Dem indicators might perform differently than Polity V, particularly given the high missingness reported for governance features.

**12. The SPEI citation is incomplete.**

The manuscript uses SPEI-3 but does not cite the original SPEI methodology paper (Vicente-Serrano et al., 2010) or specify which SPEI dataset version was used. Given that SPEI has been recalculated with updated input data, version specificity matters for reproducibility.

---

## Summary Recommendation

The manuscript makes a genuine methodological contribution through its ordinal-aware benchmark, ablation robustness testing, and SHAP temporal decomposition. However, the literature review is approximately 3-5 years behind the current state of the field. The authors appear to have conducted their literature search around 2020-2022 and not updated it for subsequent developments. For a journal submission in 2026, this is a serious deficiency. The competing datasets omission (Major Concern 3) and the inadequate characterization of the WPS tool (Major Concern 2) are particularly damaging because they undermine the manuscript's claim to establish a benchmark: a benchmark requires positioning against all relevant alternatives, not just one's own prior work.

I recommend **major revision** with a substantially updated literature review, explicit positioning against competing datasets and recent ML-for-conflict methodologies, and acknowledgment of the TFDD's institutional evolution. The core analytical contribution is sound and publishable, but the framing must reflect the field as it exists today.

---

*Reviewer 8, March 2026*
