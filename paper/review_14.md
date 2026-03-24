# Peer Review 14: Writing Quality and Stylistic Assessment

**Reviewer**: Peer Reviewer 14 (Senior Science Editor, expertise in scientific prose and Nature/GEC house style)

**Manuscript**: "Predicting Transboundary Water Conflict Outcomes: An Ordinal Machine Learning Benchmark on the TFDD"

---

## General Assessment

This manuscript is ambitious in scope and technically rich, but its prose frequently works against the reader rather than with them. The writing oscillates between lucid, well-paced exposition and dense, clause-laden passages that demand multiple readings to parse. For a Global Environmental Change audience, which spans political scientists, geographers, climate researchers, and policy practitioners alongside quantitative modellers, the current draft leans too heavily toward ML-specialist conventions at the expense of accessibility. The paper reads as roughly 70% coherent narrative and 30% technical catalogue, and the balance needs to shift.

Below I provide section-by-section analysis with specific line-level recommendations.

---

## Abstract

The abstract attempts to compress the entire results section into a single paragraph and suffers for it. At approximately 220 words, it is within journal limits but packs in so many quantitative details (QWK values, percentages, feature counts, confidence intervals) that the policy-relevant message is buried. A reader unfamiliar with QWK will encounter it three times before any definition is offered. The phrase "ordinal-aware machine learning benchmark" in the opening sentence is jargon-dense; a GEC reader needs to understand why ordinality matters before being told the study is "ordinal-aware."

**Specific suggestions:**

- Open with the policy stakes, not the methodological framing. The current first sentence ("Transboundary water interactions shape geopolitical stability for billions of people, yet quantitative prediction of conflict outcomes remains underdeveloped") is strong but is immediately followed by dense method description. Insert one sentence explaining why ordinal prediction matters (the difference between mild disagreement and militarised confrontation) before diving into technical details.
- The parenthetical "(cooperation momentum, prior event counts)" will mean nothing to most readers at this stage. Remove or replace with a plain-language equivalent such as "features derived from past conflict history."
- "Inflates validation QWK to 0.502 but does not improve test generalization" is good, clear writing. More of this throughout.
- The final sentence about spatial cross-validation is effective as a standalone conclusion.

---

## Introduction

The introduction is the strongest section stylistically. It builds a clear intellectual narrative from the "water wars" thesis through the hydro-hegemony framework to the TWINS model, and lands on a well-articulated research gap. The prose is generally well-paced and accessible. However, two problems emerge.

First, the second paragraph (lines 25-26) is a single block of approximately 280 words that attempts to survey the entire conceptual toolkit of hydropolitics in one breath. By the time the reader reaches Warner and Zawahri (2012) at the end, the thread connecting each citation to the central argument has frayed. I recommend splitting this paragraph after the TWINS discussion (Mirumachi and Allan, 2007), creating one paragraph on power and interaction frameworks and a second on institutional resilience and treaty design. This would also create a natural bridge to the institutional-change hypothesis that becomes central in the Discussion.

Second, the final paragraph of the introduction (line 31) uses a numbered list structure ("First... Second... Third...") that works well for clarity but ends abruptly. The phrase "revealing both the promise and the limitations of current feature sets for this task" is a weak closer for an introduction. A stronger final sentence would forecast the key finding, for example, that institutional dynamics outperform climate and governance indicators as predictors, giving the reader a reason to continue.

One awkward phrasing: "an omission that risks overfitting and obscures the true drivers of conflict dynamics" (line 29) conflates two different problems (statistical overfitting and substantive misinterpretation) in a way that may confuse non-ML readers. Separate these into distinct clauses.

---

## Methods

The Methods section is competent but reads like a technical manual rather than a narrative. Each subsection is self-contained, which aids reference but impedes flow. The transitions between subsections are essentially absent; the reader jumps from "Data sources and integration" to "Feature engineering" to "Target formulation" without connective tissue explaining why each step follows from the previous one.

**Specific issues:**

- The "Data sources and integration" subsection (lines 37-38) is a single long sentence listing eight data sources separated by commas. This is extremely difficult to parse. Consider a brief enumerated list or, better, group the sources by domain (climate, economic, governance, hydrological, institutional) with one sentence per group.
- "Target formulation" (lines 47-48) contains a 150-word parenthetical that defines each BAR class boundary. This is essential information but is buried in a parenthetical clause within an already long sentence. Extract this into its own paragraph or a small table.
- The phrase "An important caveat" opening the Target formulation subsection is welcome and demonstrates good epistemic practice, but the caveat itself runs to approximately 100 words before the main point of the subsection (the 4-class grouping) is reached. Lead with the grouping, then present the caveats.
- "We adapted an incremental forward selection paradigm (Guyon & Elisseeff, 2003)" in the Ablation protocol subsection uses "paradigm" where "procedure" or "approach" would be more precise and less grandiose.
- The validation strategy subsection (lines 55-56) is refreshingly concise. More of the Methods should adopt this density.
- Technical terms are generally well-defined on first use. "Quadratic weighted kappa" is introduced with a clear functional explanation ("penalises predictions proportionally to the squared distance from the true ordinal class"). "SHAP" is expanded on first use. "Ablation" is used without definition, however, and will be unfamiliar to many GEC readers. A one-sentence gloss (systematically adding or removing groups of variables to measure their contribution) would help.

---

## Results

The Results section is where writing quality diverges most sharply. The subsection "Systematic ablation identifies candidate predictive feature groups" (lines 72-101) is well-structured: it presents primary results, acknowledges path-dependence, and uses tables effectively. The writing is direct and appropriately hedged ("candidate predictive feature groups" rather than "key predictors").

However, "Model comparison and the autoregressive endogeneity finding" (lines 103-120) becomes syntactically congested. The sentence beginning "This pattern, where autoregressive features improve in-sample performance at the expense of out-of-sample generalization, confirms that Optuna's search exploits the endogenous signal aggressively" (line 118) is 30 words long and requires the reader to hold a subordinate clause in memory while processing a technical claim. Break this into two sentences.

The sentence at line 118 that begins "The autoregressive inflation also interacts with the QWK-versus-macro-F1 tradeoff" introduces a new concept ("QWK-versus-macro-F1 tradeoff") that has not been established and immediately follows it with a parenthetical containing a specific prediction count ("343 predicted vs 34 actual on test"). This is too much cognitive load in one sentence.

The SHAP analysis subsection (lines 122-126) is clear and well-paced. The temporal decomposition subsection (lines 128-132) effectively links quantitative results to substantive theory. The sentence "treaty formation rate increased 56.5% in importance from the Cold War era to the post-2000 period" is precise and readable.

The geographic concentration subsection (line 136) contains a single paragraph of approximately 200 words that attempts to cover global patterns, specific basin examples, a methodological caveat about the BAR scale, and a temporal comparison. This paragraph needs to be divided into at least two units: one on spatial patterns, one on the BAR scale limitation. The parenthetical "(59.3% conflict ratio, 27 events)" pattern, repeated three times in sequence, creates a staccato rhythm that interrupts the narrative flow. Consider moving basin-specific numbers to a supplementary table and summarising the pattern in prose.

---

## Discussion

The Discussion is the most uneven section. Its four-part structure (institutional velocity, economic capacity, climate, geopolitical regime dependence) is clear and logical, and the opening epistemological caveat (lines 158-159) is exemplary. The language ("consistent with," "suggests," "associational hypotheses") is appropriately cautious throughout.

However, several passages suffer from redundancy. The climate subsection (lines 177-185) contains two subsection headers for what is effectively one discussion point, which reads as a drafting artefact. The first header, "Climate as a structural mediator, not a proximate trigger," states a conclusion, while the second, "Climate data specification does not improve event-level prediction at this resolution," states a finding. Choose one. The subsection then runs to approximately 400 words and repeats points already made in the Results ("delta QWK of -0.042 under the primary ordering," "grouped permutation importance yielded a positive contribution of +0.019"). A Discussion section should interpret results, not restate them.

The sentence at line 181 beginning "The delta QWK of -0.042 under the primary ordering is consistent with the meta-analytic finding of Mach et al. (2019)..." runs to approximately 60 words and contains three subordinate clauses. This is the kind of sentence that causes a reader to lose the thread. Break it at the comma after "dominating."

The governance data gaps subsection (lines 193-197) is the most effective part of the Discussion: concise, novel, and policy-relevant. It demonstrates what the entire Discussion could be if tightened.

---

## Limitations

The limitations section (lines 200-201) is a single paragraph of approximately 350 words containing ten distinct limitations connected by "First... Second... Third..." This is exhaustive but exhausting. Consider grouping related limitations (e.g., data limitations together, methodological limitations together, generalisability limitations together) into separate short paragraphs with topic sentences. The current format reads as a defensive catalogue rather than a thoughtful reflection.

---

## Conclusion

The conclusion (lines 203-211) effectively synthesises the four policy insights and two methodological contributions. The numbered structure works well here. The final paragraph addressed to "the water governance community" is a strong closer. One concern: the sentence beginning "Future work should extend this framework..." lists four distinct directions separated by commas. At 50+ words, it asks the reader to hold too much in working memory. Use a brief enumerated list or break into two sentences.

---

## Passive Voice

Passive voice is used appropriately and not excessively. Most methodological descriptions use passive constructions ("We constructed features," "We enforced strict temporal splitting") that are standard and clear. The occasional agentless passive ("investment in institutional velocity... deserves priority") is effective for policy-directed statements. No revision needed on this axis.

---

## Figure and Table Captions

Table captions are functional but could be more informative. Table 1's caption states "LightGBM validation QWK with default hyperparameters under a fixed temporal split protocol" but does not state the temporal split boundaries, which are defined elsewhere. Table 3's caption is a model of informativeness by contrast. Figure captions are adequate. Figure 4's caption effectively summarises both panels. Figure 2's caption would benefit from stating what "all four ordinal classes" are, since the reader may have forgotten the class definitions by this point.

---

## Coherence and Overall Flow

The paper reads as a largely coherent piece with two seams. The first is between the Introduction (which tells a compelling story about hydropolitics) and the Methods (which shifts abruptly into technical specification). A bridging sentence at the end of the Introduction, or a brief paragraph at the start of Methods, connecting the conceptual framework to the analytical choices would smooth this transition. The second seam is within the Discussion's climate subsection, which as noted has a duplicated header and reads as if two drafts were merged without reconciliation.

The overall arc, from theory gap through systematic analysis to policy-relevant findings tempered by honest limitations, is sound and appropriate for GEC. With targeted tightening of the longest sentences, splitting of overloaded paragraphs, and removal of redundant quantitative restatements in the Discussion, this manuscript would read as a polished, authoritative contribution.

---

## Summary of Priority Revisions

1. **Abstract**: Reduce quantitative density; lead with policy stakes; define or remove QWK on first mention.
2. **Introduction, paragraph 2**: Split the 280-word hydropolitics survey into two thematic paragraphs.
3. **Methods, Data sources**: Break the eight-source enumeration into grouped sentences or a structured list.
4. **Methods, Target formulation**: Extract the class-boundary definitions from their parenthetical prison into a standalone paragraph or table.
5. **Methods**: Define "ablation" for non-ML readers on first use.
6. **Discussion, Climate subsection**: Remove the duplicate header; cut restated quantitative results; tighten the 60-word Mach et al. sentence.
7. **Limitations**: Group into thematic paragraphs rather than a single numbered list.
8. **Throughout**: Break sentences exceeding 40 words into shorter units. The manuscript contains at least a dozen sentences above this threshold that impede readability.
9. **Throughout**: Add brief transitional sentences between Methods subsections to maintain narrative momentum.
