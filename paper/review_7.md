# Peer Review: Predicting Transboundary Water Conflict Outcomes

**Reviewer 7** -- Global Environmental Change

**Expertise**: Hydropolitics literature, transboundary water governance theory, history of the "water wars" debate

**Review focus**: Literature engagement, theoretical framing, and citation accuracy

---

## General Assessment

This manuscript presents an ordinal machine learning benchmark on the TFDD dataset and attempts to situate its findings within the hydropolitics literature. The technical contribution is competent, and the author demonstrates familiarity with major works in the field. However, the literature engagement, while broad in citation count, is uneven in depth. Several frameworks are invoked without being fully operationalized or critically interrogated, and notable gaps exist in the coverage of recent scholarship. For a journal such as Global Environmental Change, which expects manuscripts to advance theoretical understanding alongside empirical contribution, the current framing needs strengthening. The discussion sections do a better job of connecting results to theory than the introduction, but inconsistencies between claimed theoretical grounding and actual analytical practice undermine some of the manuscript's stronger claims.

---

## Major Concerns

**1. The hydro-hegemony framework is cited but not operationalized, creating a disconnect between theory and analysis.**

Zeitoun and Warner (2006) and Zeitoun and Mirumachi (2008) are cited in the introduction as establishing that "transboundary water outcomes are shaped by asymmetric power relations rather than pure resource logic." The discussion then claims that economic indicators are "consistent with the hydro-hegemony framework." However, the hydro-hegemony framework is a multi-dimensional theory encompassing riparian position, exploitation potential, and three pillars of power (geographic, material, and bargaining/ideational). The manuscript operationalizes only material power asymmetry through GDP and military expenditure ratios. The author acknowledges the absence of discursive power, agenda-setting capacity, and "sanctioned discourse" (p. Discussion), but this acknowledgment comes too late and is too brief. A GEC submission should either (a) operationalize the framework more fully, including riparian position (upstream/downstream), which the author notes is absent, or (b) explicitly reframe the contribution as testing a narrow, material-power subset of the hydro-hegemony hypothesis. As written, the manuscript risks overstating its theoretical engagement with this framework.

**2. The TWINS framework (Mirumachi and Allan, 2007) is cited but its core insight is not integrated into the research design.**

The introduction correctly describes TWINS as demonstrating that "conflict and cooperation are not mutually exclusive but routinely coexist within the same basin, operating at different intensities simultaneously," and the discussion notes that "a single BAR score per event may capture only one dimension of a multi-dimensional interaction space." This is a fundamental critique of the entire analytical approach, yet the manuscript proceeds without addressing it. If TWINS is taken seriously, the dependent variable (a single BAR score per event) is structurally inadequate to capture simultaneous conflict-cooperation dynamics. The manuscript should either engage more substantively with why a single-score approach remains defensible despite TWINS, or acknowledge this as a foundational limitation rather than a parenthetical remark buried in the discussion. As it stands, the TWINS citation reads as decorative rather than substantive.

**3. Missing engagement with the critical hydropolitics literature and recent theoretical advances.**

Several important bodies of work are absent:

(a) Selby (2003, "Water, Power and Politics in the Middle East") and Selby and Hoffmann (2014) represent the critical hydropolitics school that challenges the empiricist-positivist approach to water conflict quantification. Given that this manuscript is squarely positivist, engaging with critiques of the quantification enterprise itself would strengthen the epistemological framing. A GEC audience will expect awareness of these debates.

(b) Shlomi Dinar's broader body of work on treaty design and negotiation theory (Dinar, 2008, "International Water Treaties") is relevant to the treaty formation rate finding but is not cited. Only the 2015 scarcity-cooperation paper is included.

(c) Tir and Stinnett (2012, "Weathering climate change: Can institutions mitigate international water conflict?") directly tests the interaction between climate variability and institutional capacity on water conflict, which is precisely the climate-governance interaction the manuscript identifies as needing future investigation. This omission is notable because the authors' own results point toward this interaction.

(d) Zawahri and Mitchell (2011) on the design and effectiveness of river treaties, and De Stefano et al. (2012, 2017) on institutional resilience and vulnerability in transboundary basins, are directly relevant to the institutional velocity finding but are absent.

(e) Hensel, Mitchell, and Sowers (2006) on conflict management of enduring rivalries over international rivers is relevant to the basin-level path dependence captured by the cooperation momentum variable.

**4. The climate-conflict literature engagement, while improved by citing the correct seminal works, lacks critical depth.**

The manuscript cites Mach et al. (2019), Buhaug (2010), Koubi (2019), Ide (2015), and Ide et al. (2020). However, the engagement is largely referential rather than analytical. For instance, the Buhaug (2010) citation is reduced to showing that "climate-conflict statistical relationships are highly sensitive to model specification," which was only one element of that paper's contribution; its core argument about the insufficiency of environmental determinism for explaining African civil wars is not discussed. Similarly, Koubi (2019) is cited as a general review but the specific mechanisms she identifies (agricultural income shocks, migration, strategic resource competition) are not mapped onto the feature set. The Mach et al. (2019) expert elicitation, which found that climate's contribution to conflict risk is judged "low" to "moderate" by experts, is mentioned only in terms of its 3-20% range estimate. The nuance of that study, that expert confidence increases substantially when institutional and governance mediators are included, would directly support the manuscript's own governance-interaction argument.

**5. The Discussion does not adequately connect the spatial cross-validation results back to hydropolitics theory.**

The finding that model performance varies dramatically across continents (QWK 0.062 in North America to 0.417 in Europe) is presented primarily as a methodological limitation. But this pattern has theoretical significance: it may reflect the varying institutional architectures of different continents (Europe's dense treaty network via UNECE Water Convention vs. North America's bilateral ad hoc arrangements). The discussion of North America's anomalous conflict ratio mentions "institutionalised US-Mexico/Canada disagreements" but does not connect this to the broader literature on how different institutional architectures produce different conflict-cooperation dynamics. Zeitoun and Mirumachi (2008), Warner and Zawahri (2012), and the institutional resilience literature (Giordano and Wolf, 2003) all provide frameworks for interpreting these geographic differences theoretically rather than treating them as noise.

---

## Minor Concerns

**6. The Basins at Risk (BAR) framework attribution needs clarification.**

The manuscript attributes the BAR framework to both "Wolf et al. (2003)" and "Yoffe, Wolf & Giordano (2003)" at different points, and it is unclear whether these refer to the same or different publications. The reference list contains both as separate entries (Wolf, Yoffe & Giordano 2003 in Water Policy; Yoffe, Wolf & Giordano 2003 in JAWRA), but the text does not distinguish their respective contributions clearly. The BAR scale itself originates in the earlier work, and the two 2003 papers make distinct contributions. The manuscript should be precise about which paper contributes which element.

**7. The Gleick (1993) citation is slightly mischaracterized.**

The manuscript states the "water wars" thesis was "popularised by early environmental security scholarship (Gleick, 1993; Homer-Dixon, 1999)." However, Gleick's 1993 paper in International Security is more nuanced than the "water wars" label implies; he explicitly discussed conditions under which water might or might not lead to conflict and did not simply predict inevitable water wars. Homer-Dixon (1999) similarly argued for indirect pathways through environmental scarcity rather than direct resource wars. The "water wars" label is more fairly attributed to journalistic and popular accounts (e.g., Starr 1991, Bulloch and Darwish 1993) than to these academic works. This matters because the manuscript frames the entire literature trajectory as a correction of naive water wars predictions, when the academic literature was always more sophisticated than this framing suggests.

**8. Allan (2001) is invoked in the climate discussion but his primary contribution, virtual water theory, is not mentioned.**

The citation reads: "Allan (2001) argued that water scarcity operates as a slow-moving stressor whose effects are mediated through political and economic institutions." While Allan did argue this, his central contribution in the cited book was the concept of virtual water and its role in alleviating physical water scarcity through trade. This is directly relevant to the manuscript's economic capacity findings, since virtual water imports require economic capacity, but the connection is not drawn.

**9. The Bernauer and Bohmelt (2020) citation should be verified.**

The reference list gives "Bernauer, T. & Bohmelt, T. International water cooperation and interstate conflict. Global Environmental Change 65, 102161 (2020)." I believe the correct title and framing of this work should be double-checked. Additionally, the umlaut in Bohmelt should be rendered as Bohmelt or Boehmelt consistently, and the correct spelling is Bohmelt with an umlaut (Bohmelt). GEC has specific formatting requirements for diacritical marks.

**10. The WPS Partnership is cited without a formal reference.**

The manuscript discusses the Water Peace and Security Partnership's operational tool, including specific performance metrics (86% capture rate, 50% false positive rate), but provides no formal citation. For a GEC submission, operational tools and their performance claims require proper referencing to verifiable sources.

**11. The Conca, Wu & Mei (2006) engagement could be strengthened.**

This work is cited twice: once in the introduction for the claim that "participatory design processes generating more durable agreements than top-down frameworks," and once in the discussion. However, the specific relevance to the manuscript, that the institutional architecture of treaty-making (which could in principle be coded from TFDD treaty metadata) shapes compliance, is not explored as a potential feature engineering direction.

**12. Missing engagement with the broader environmental peacebuilding literature.**

The manuscript focuses narrowly on conflict prediction but does not engage with the environmental peacebuilding literature (Ide, 2019; Swain and Ojendal, 2018; Harari and Rosendahl, 2022), which reframes shared water resources as potential catalysts for cooperation rather than conflict. Given that 77% of events in the dataset are cooperative, this literature is directly relevant to interpreting the class imbalance and the predominance of cooperative outcomes.

**13. The framing of "first ordinal-aware ML benchmark" should be qualified.**

The abstract claims this is "to our knowledge, the first ordinal-aware machine learning benchmark on the TFDD." While this may be technically accurate for the specific combination of ordinal methods on TFDD event-level data, the claim should acknowledge prior quantitative work on the BAR scale, including Yoffe et al. (2004) and the various BAR-derived analyses that have used the ordinal structure implicitly.

---

## Summary Recommendation

The manuscript makes a legitimate empirical contribution by applying ordinal-aware ML with systematic ablation to the TFDD. However, the literature engagement falls short of GEC standards in several respects: key frameworks are cited but not operationalized, the critical hydropolitics literature is absent, and the discussion does not fully exploit the theoretical implications of its own findings. The climate-conflict literature is adequately cited but superficially engaged. I recommend major revisions focused on (1) deepening the engagement with hydro-hegemony and TWINS beyond citation to operationalization or explicit delimitation, (2) incorporating the missing seminal works identified above, (3) strengthening the theoretical interpretation of the spatial cross-validation findings, and (4) ensuring all citations are accurate and properly attributed. The manuscript has the ingredients of a strong GEC contribution, but the current version reads more as a methods paper with a literature survey than as a theory-informed empirical analysis.

**Recommendation: Major Revision**
