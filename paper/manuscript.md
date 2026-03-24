# Predicting Transboundary Water Conflict Outcomes: An Ordinal Machine Learning Benchmark on the TFDD

**Authors**: Felix Meier

**Affiliations**: Macquarie University & Australian Genome Foundry, Sydney, Australia

**Correspondence**: felix.meier@mq.edu.au

**Keywords**: transboundary water conflict, machine learning, ordinal classification, hydropolitics, conflict prediction, SHAP explainability

---

## Abstract

Quantitative prediction of transboundary water conflict outcomes remains underdeveloped, with prior approaches neglecting the ordinal structure of the Basins at Risk (BAR) scale, lacking systematic feature evaluation, and providing limited interpretability. Here we present an ordinal machine learning benchmark on the Transboundary Freshwater Dispute Database (TFDD), analysing 6,805 events across 313 international river basins (1948-2008) with eight external data sources. Our primary methodological contributions are three diagnostic findings with broad relevance to conflict prediction. First, autoregressive features derived from prior basin events (cooperation momentum, event counts) inflate validation performance by 0.10 QWK without improving test generalization (test QWK 0.298 without vs 0.290 with), exposing a common but underreported source of endogenous information leakage. Second, sequential feature ablation is path-dependent: three orderings of seven feature groups produce three different retained sets, demonstrating that incremental ablation cannot establish definitive feature importance rankings. Third, event-level prediction does not generalize reliably across temporal regimes (validation-to-test gap of 26%) or geographic contexts (continent-level spatial CV: QWK 0.062 to 0.417), establishing sober baseline expectations for the field. From a policy perspective, exploratory SHAP analysis generates testable hypotheses: the rate of treaty formation may matter more than treaty stock for conflict prevention, economic capacity is consistently associated with cooperative outcomes, and basin-averaged annual climate indicators at 0.5-degree resolution show ambiguous incremental value consistent with climate operating as a structural mediator rather than a proximate trigger. These policy-relevant associations require causal confirmation through quasi-experimental or instrumental variable designs.

---

## Introduction

Freshwater scarcity is intensifying as a geopolitical pressure point. Approximately 2.4 billion people live under conditions of water stress, and 286 river basins cross international boundaries, generating persistent interdependencies that neither upstream nor downstream states can unilaterally resolve (Wolf et al., 2003). While 77% of recorded transboundary water interactions are cooperative, conflict events have accelerated since 2017, and the consequences of prediction failure are severe: disrupted agriculture, displaced populations, and in extreme cases, militarised confrontation.

The "water wars" thesis, popularised by early environmental security scholarship (Gleick, 1993; Homer-Dixon, 1999), posited that resource scarcity would drive interstate violence. Wolf and colleagues systematically challenged this framing, demonstrating through the Basins at Risk (BAR) framework that cooperation overwhelmingly dominates the historical record, and that rapid institutional or physical change, rather than scarcity per se, best explains the transition to conflict (Yoffe, Wolf & Giordano, 2003; Wolf et al., 2003). This insight reframed the field: the question shifted from whether water wars would occur to under what conditions cooperative or conflictual outcomes emerge. Crucially, however, Dinar, Katz & Shmueli (2015) showed that scarcity can itself stimulate institutional innovation, with water-scarce states more likely to negotiate adaptive agreements, a dynamic that further complicates simple scarcity-to-conflict narratives.

The hydropolitics literature has since developed a sophisticated conceptual toolkit. Zeitoun and Warner (2006) established the hydro-hegemony framework, showing that transboundary water outcomes are shaped by asymmetric power relations rather than pure resource logic, while Zeitoun and Mirumachi (2008) extended this into a systematic analysis of how hegemonic structures condition the range of possible outcomes in shared basins. Mirumachi and Allan (2007) introduced the Transboundary Waters Interaction Nexus (TWINS), demonstrating that conflict and cooperation are not mutually exclusive but routinely coexist within the same basin, operating at different intensities simultaneously. This insight is directly relevant to prediction: a binary conflict/cooperation framing misses the ordinal gradations that the BAR scale was designed to capture. Cascao (2009) extended this understanding by examining counter-hegemonic strategies in the Nile basin, showing that weaker riparian states are not passive actors but deploy coalition-building and norm entrepreneurship to shift negotiation dynamics. Giordano and Wolf (2003) demonstrated that institutional resilience, the capacity of existing agreements to absorb new stresses, is a stronger predictor of cooperative outcomes than physical water availability, while Brochmann and Hensel (2009) showed that institutionalised management mechanisms significantly reduce the probability of militarised disputes over shared rivers. Conca, Wu & Mei (2006) demonstrated that the institutional architecture of environmental treaty-making shapes long-term compliance and adaptive capacity, with participatory design processes generating more durable agreements than top-down frameworks. Bernauer and Bohmelt (2020) provided rigorous econometric evidence that institutional design characteristics, particularly provisions for joint monitoring and dispute resolution, predict treaty effectiveness independently of the political will of signatories. Warner and Zawahri (2012) further showed that power asymmetries shape not only treaty content but compliance patterns, with hegemonic states selectively enforcing provisions that serve their interests.

Despite this conceptual advance, quantitative prediction of transboundary water conflict outcomes remains limited. Ge et al. (2022) applied boosted regression trees to predict conflict onset at basin scale, finding climate sensitivity to be a significant driver. The Water Peace and Security (WPS) Partnership developed a random forest and LSTM pipeline achieving 86% capture rate for conflict events but with a 50% false positive rate, limiting operational utility. Jiang et al. (2025) proposed a water dependency framework explaining 80% of historical conflicts through structural vulnerability. Unfried et al. (2022) used GRACE satellite-derived water mass anomalies as instrumental variables, establishing causal links between water availability and localised conflict. However, these approaches share critical limitations: none account for the ordinal structure of the BAR scale, most lack systematic feature selection, and few provide interpretable explanations for their predictions.

The BAR scale, ranging from -7 (formal war) to +7 (voluntary unification into one nation), encodes a 14-point ordinal gradient of conflict intensity. Treating this as a binary classification or continuous regression discards meaningful structure. Furthermore, prior studies have not systematically tested which data sources contribute predictive power and which introduce noise, an omission that risks overfitting and obscures the true drivers of conflict dynamics. This gap is consequential: if practitioners cannot distinguish which data types reliably shift model performance, investment in monitoring infrastructure remains unguided.

We address these gaps with four contributions. First, we construct, to our knowledge, the first ordinal-aware ML benchmark on the TFDD, integrating eight external datasets and evaluating ordinal classification on the BAR scale. Second, we expose critical methodological pitfalls in conflict prediction through three diagnostic findings: autoregressive features derived from prior basin events inflate validation metrics without improving test generalization, sequential feature ablation is path-dependent (different orderings retain different feature groups from the same candidate pool), and governance indicator missingness (60-64%) likely causes feature group evaluation artifacts under median imputation. Third, we demonstrate that event-level water conflict prediction does not generalize reliably across temporal regimes or geographic contexts, establishing baseline expectations for the field. Fourth, we derive tentative policy-relevant hypotheses about institutional velocity, economic capacity, and climate scale-mismatch that merit causal investigation.

## Methods

### Data sources and integration

The primary dataset is the TFDD, maintained by Oregon State University, containing 6,805 coded water interaction events across 313 international river basins from 1948 to 2008. Each event is scored on the BAR scale from -7 to +7. We enriched these events with eight external sources: CRU TS 4.09 gridded climate data (0.5-degree resolution monthly precipitation and potential evapotranspiration), the Standardised Precipitation-Evapotranspiration Index at 3-month scale (SPEI-3) for drought characterisation, World Bank World Development Indicators (GDP per capita, total population, military expenditure as percentage of GDP), Worldwide Governance Indicators (rule of law, political stability), Polity V democracy scores, FAO AQUASTAT national water resource statistics (total renewable water resources, agricultural water withdrawal percentage, water dependency ratio), and TFDD's own treaty database (3,812 treaties) and spatial database (818 basin-country units, 2024 update).

### Feature engineering

We constructed features at the basin-country unit (BCU) level, the natural unit of analysis for transboundary interactions. Climate variables were aggregated zonally across basin polygons using area-weighted means. Economic and governance indicators were matched at the country-year level for each state party in the event. For multi-country events, we computed both mean values and asymmetry ratios (e.g., GDP ratio between the wealthiest and poorest participating states, dam count ratios, withdrawal ratios, institutional quality differentials).

Treaty-related features required careful handling to avoid data leakage. We computed cumulative treaty counts strictly prior to each event date, along with treaty formation rates (treaties per year in the preceding 5-year and 10-year windows). Temporal dynamics features captured cooperation momentum (rolling mean BAR of prior events in the same basin), event escalation (count and intensity trend of events in the preceding 5 years), and era indicators (Cold War binary, post-2000 binary).

### Target formulation

An important caveat: the BAR scale was designed as a descriptive coding instrument for the TFDD, not as a dependent variable for predictive modelling. Its ordinal structure assumes meaningful spacing between adjacent values, but the political consequence of moving from BAR -1 (mild verbal hostility) to -2 (official diplomatic protest) differs qualitatively from moving from -5 (small-scale military action) to -6 (extensive war). The use of quadratic weighted kappa as a metric implicitly weights ordinal distance, an assumption we adopt but acknowledge is imperfect. Inter-coder reliability statistics for BAR scoring are not reported in the original TFDD documentation, introducing unmeasured noise into the target variable. With these caveats, we grouped the 14-point BAR scale into four ordinal classes reflecting the natural thresholds identified by Yoffe, Wolf & Giordano (2003) in their original BAR coding work: negative interactions (BAR < 0; 19.1% of events, hereafter "conflict" for brevity, though this category encompasses diplomatic disputes, verbal hostility, and economic sanctions, not exclusively armed conflict), neutral (BAR = 0; 4.0%), mild cooperation (0 < BAR <= 3; 51.9%), and strong cooperation (BAR > 3; 24.9%). The conflict/neutral boundary at BAR = 0 is the most theoretically grounded threshold in the BAR framework, separating negative from positive interactions. The mild/strong cooperation boundary at BAR = 3 separates events corresponding to bilateral working agreements and technical exchanges (BAR 1-3) from multilateral accords, joint institutional arrangements, and voluntary integration outcomes (BAR 4-7), reflecting a qualitative shift in the depth and scope of cooperative engagement. Sensitivity analyses using 3-class and 5-class groupings are reported in Extended Data.

### Ablation protocol

We adapted an incremental forward selection paradigm (Guyon & Elisseeff, 2003) to systematically evaluate each feature group. The protocol was fixed across all tests: temporal train/validation split, LightGBM classifier with default hyperparameters, and QWK as the primary metric. Starting from a baseline of 29 TFDD-intrinsic features (basin attributes, treaty counts, event metadata), we added each feature group sequentially and measured the change in validation QWK (Table 1). A feature group was retained only if it improved QWK by at least 0.005; otherwise it was discarded. Because sequential ablation is inherently order-dependent, we tested three alternative orderings (original, reversed, and shuffled) and computed 1,000-resample bootstrap confidence intervals on QWK deltas to assess the statistical robustness of retention decisions (Table 2). Additionally, we report permutation importance across all retained features as an order-independent validation. The primary ordering yielded a final set of 45 features (baseline + economic + temporal), pruned from an initial pool of 82 candidates, though the robustness analysis revealed that specific retention decisions are sensitive to group ordering (see Results).

### Validation strategy

We enforced strict temporal splitting to simulate prospective forecasting: training data comprised events before 1996, validation data covered 1996 to 2002, and the held-out test set spanned 2003 to 2008. This temporal separation prevents information leakage from future events into model training. Hyperparameter tuning used basin-grouped 5-fold cross-validation within the training set to prevent within-basin information leakage during optimisation.

### Models

We evaluated six models spanning two families. Ordinal regression baselines included LogisticAT (all-thresholds ordinal logistic regression) and OrdinalRidge (ridge-penalised ordinal regression). Gradient-boosted tree models included LightGBM and XGBoost, each tested with default hyperparameters and after 100-trial Optuna Bayesian optimisation. All gradient-boosted models used the ablation-selected 45-feature set.

### Evaluation metrics

The primary metric was quadratic weighted kappa (QWK), which penalises predictions proportionally to the squared distance from the true ordinal class, making it well-suited for ordinal outcomes. Secondary metrics included macro-averaged F1 score (to assess per-class balance) and overall accuracy. We computed 95% bootstrap confidence intervals (1,000 resamples) for QWK on the validation set using percentile bootstrap, and applied McNemar's test for pairwise model comparison (Extended Data Figure 3). These bootstrap CIs resample predictions from a fixed model, capturing metric estimation uncertainty but not model training uncertainty; fold-level variance from the nested CV procedure provides a complementary estimate of model instability.

### Explainability

We applied SHAP (SHapley Additive exPlanations) TreeExplainer to the best-performing model. Global feature importance was measured as mean absolute SHAP value across all classes. For temporal decomposition, we partitioned events into three eras (Cold War: pre-1990; post-Cold War: 1990-1999; post-2000: 2000-2008) and computed era-specific mean absolute SHAP values, measuring the percentage change in each feature's importance across eras.

## Results

### Systematic ablation identifies candidate predictive feature groups

The ablation protocol revealed asymmetries in the predictive value of different data sources, though robustness testing demonstrated that specific retention decisions are sensitive to feature group ordering (Tables 1 and 2).

Under the primary ordering, the baseline TFDD feature set (basin attributes, treaty counts, event metadata; 29 features) achieved a validation QWK of 0.353. Adding climate variables (precipitation, PET, SPEI-3 drought index, anomalies; 5 additional features) decreased performance by 0.042. Governance indicators (Polity V, WGI rule of law, political stability) produced a negligible change of -0.003. Economic indicators (GDP per capita, military expenditure, total population, water withdrawal) yielded a marginal gain of +0.003, below the 0.005 retention threshold. AQUASTAT hydrological variables met the threshold (+0.005), and asymmetry features provided the largest gain (+0.046). Temporal dynamics features reduced performance (-0.012) after asymmetry was already retained. Under this primary ordering, the retained set was baseline + AQUASTAT + asymmetry (43 features).

However, robustness analysis across three orderings revealed substantial path-dependence (Table 2). The reversed ordering retained baseline + temporal + economic + climate (49 features), while the shuffled ordering retained baseline + economic + temporal (45 features). Climate variables, discarded under the primary and shuffled orderings, were retained under the reversed ordering. Economic features, retained only when entered before governance and climate, were discarded under the primary ordering where they followed governance. These results demonstrate that sequential ablation identifies candidate feature groups whose incremental value depends on what has already been included, rather than establishing definitive feature importance rankings.

Permutation importance, which is order-independent, identified cooperation momentum (0.042), events in the prior 5 years (0.027), bilateral indicator (0.024), and number of countries (0.017) as the features with the largest marginal contributions to validation QWK.

**Table 1. Ablation results under primary ordering.** LightGBM validation QWK with default hyperparameters under a fixed temporal split protocol. Delta QWK computed relative to the best retained configuration at each step.

| Feature Group | n | QWK | Delta | Decision |
|:---|:---:|:---:|:---:|:---:|
| Baseline TFDD | 29 | 0.353 | -- | RETAIN |
| +Climate | 34 | 0.311 | -0.042 | DISCARD |
| +Governance | 39 | 0.350 | -0.003 | DISCARD |
| +Economic | 39 | 0.356 | +0.003 | DISCARD |
| +AQUASTAT | 35 | 0.358 | +0.005 | RETAIN |
| +Asymmetry | 43 | 0.404 | +0.046 | RETAIN |
| +Temporal | 49 | 0.393 | -0.012 | DISCARD |

**Table 2. Ablation path-dependence: retained groups vary across orderings.**

| Ordering | Retained Groups | Final n | Val QWK |
|:---|:---|:---:|:---:|
| Primary (climate first) | baseline + AQUASTAT + asymmetry | 43 | 0.404 |
| Reversed (temporal first) | baseline + temporal + economic + climate | 49 | 0.400 |
| Shuffled (economic first) | baseline + economic + temporal | 45 | 0.412 |

Feature group definitions: Baseline TFDD: basin attributes, treaties, event metadata. Climate: precipitation, PET, SPEI-3 drought, anomalies. Governance: Polity V, WGI rule of law, political stability. Economic: GDP/capita, military spend, population, water withdrawal. AQUASTAT: water dependency ratio, agricultural withdrawal %. Asymmetry: GDP ratio, dam ratio, withdrawal ratio, institutional differential. Temporal: event escalation, cooperation momentum, Cold War indicator, treaty formation rate.

### Model comparison and the autoregressive endogeneity finding

We report model comparison on two feature sets: the full 45-feature ablation-selected set (baseline + economic + temporal, including autoregressive features) and the 42-feature non-autoregressive subset that excludes cooperation momentum, events in the prior 5 years, and event escalation. Three of the 45 features are mechanically correlated with the target variable since they are derived from the same BAR scale used for labelling, creating endogeneity risk. As shown in Table 3, removing these features consistently improves test-set performance across both default and Optuna-tuned models, despite reducing validation QWK. We therefore adopt the 42-feature non-autoregressive model as our primary result.

**Table 3. Model comparison.** The 42-feature non-autoregressive model (primary) and 45-feature model (for comparison). 95% bootstrap CIs on validation set (1,000 resamples). Optuna uses nested 5-fold basin-grouped CV.

| Model | Features | Val QWK [95% CI] | Test QWK | Val F1 | Test F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Majority class baseline | -- | 0.000 | -- | 0.170 | -- |
| LogisticAT (tuned) | 42 | 0.174 [0.139, 0.210] | -- | 0.235 | -- |
| OrdinalRidge (tuned) | 42 | 0.285 [0.251, 0.319] | -- | 0.218 | -- |
| XGBoost default | 42 | 0.407 | 0.146 | 0.413 | 0.290 |
| **XGBoost tuned (primary)** | **42** | **0.403** | **0.298** | **0.311** | **0.320** |
| XGBoost tuned + AR features | 45 | 0.502 [0.461, 0.539] | 0.290 | 0.373 | 0.281 |

The primary 42-feature tuned model achieves test QWK of 0.298 and test macro-F1 of 0.320. Including autoregressive features inflates validation QWK by nearly 0.10 (from 0.403 to 0.502) but actually degrades test QWK (from 0.298 to 0.290) and test macro-F1 (from 0.320 to 0.281). This pattern, where autoregressive features improve in-sample performance at the expense of out-of-sample generalization, confirms that Optuna's search exploits the endogenous signal aggressively. The autoregressive inflation also interacts with the QWK-versus-macro-F1 tradeoff: the 45-feature tuned model achieves higher validation QWK by learning to redistribute predictions across classes (over-predicting the neutral class: 343 predicted vs 34 actual on test, Extended Data Table 2) rather than improving per-class accuracy.

Per-class performance on the test set reveals an important asymmetry. The conflict class (BAR < 0) achieved recall of approximately 49.4%, indicating moderate capture of conflict events. The neutral class performed poorly (precision 0.020, recall 0.206), consistent with its very small sample size (4.0% of events, only 34 test samples). Per-class precision, recall, and F1 scores are reported in Extended Data Table 2.

### SHAP analysis reveals institutional and temporal drivers

Global SHAP analysis of the XGBoost model identified five dominant predictors (Fig. 2a). The number of countries involved in the event ranked first (mean |SHAP| = 0.373), reflecting the combinatorial complexity of multilateral negotiations. Events in the prior 5 years ranked second (0.319), capturing basin-level activity intensity. Cooperation momentum ranked third (0.318), though as shown in Table 4, both of these autoregressive features carry endogeneity risk and their high SHAP importance should be interpreted cautiously. Year of occurrence ranked fourth (0.272), encoding broad secular trends. Issue type ranked fifth (0.204), distinguishing water quantity, quality, hydropower, and navigation disputes.

No climate variable appeared among the top 15 features by SHAP importance, consistent with the ablation finding that basin-averaged annual climate data do not improve event-level prediction at this resolution. Economic features, particularly GDP per capita and military expenditure, occupied ranks 6 through 10 (Fig. 2b).

### Temporal decomposition reveals shifting driver importance

SHAP temporal decomposition across three geopolitical eras (Cold War: pre-1990; post-Cold War: 1990-1999; post-2000: 2000-2008) uncovered significant shifts in driver importance (Fig. 3). Treaty formation rate increased 56.5% in importance from the Cold War era to the post-2000 period, consistent with the hypothesis that the rate of institutional change, rather than institutional stock, drives conflict resolution (Wolf et al., 2003). Cooperation momentum decreased 26.0% in importance post-2000, suggesting that historical path dependence erodes as new actors and issues enter basin-level negotiations. The stock of treaties prior to an event decreased 19.5% in importance post-2000, further reinforcing the distinction between institutional rate and stock (Fig. 3).

These temporal shifts align with observed geopolitical dynamics. The post-Cold War period saw a proliferation of new treaty-making in previously deadlocked basins (e.g., the Mekong River Commission, 1995), elevating the salience of institutional velocity. Simultaneously, the entry of non-state actors and transboundary environmental organisations diluted the explanatory power of historical bilateral cooperation patterns.

### Geographic concentration of conflict

Conflict events were highly spatially concentrated (Fig. 4). The top 10 basins accounted for 81.3% of all conflict events in the dataset. The Jordan basin exhibited the highest conflict ratio at 40.1% (proportion of events with BAR < 0), followed by the Tigris-Euphrates at 39.6% and the Indus at 36.8%. Conversely, high-activity cooperative basins included the Danube, Mekong, La Plata, and Niger. At the continental level, North America exhibited the highest conflict ratio at 37.7%, followed by Asia at 26.3%. Disaggregation reveals that North America's high conflict ratio is driven by the Nelson-Saskatchewan (59.3% conflict ratio, 27 events), Colorado (52.2%, 23 events), and Rio Grande (48.4%, 31 events) basins, reflecting recurring US-Mexico water allocation disputes and US-Canada disagreements coded at comparable BAR intensity to Middle Eastern water conflicts despite occurring within substantially stronger institutional frameworks. This finding highlights a limitation of the BAR scale for cross-regional comparison: institutionalised disagreements within robust treaty regimes may receive similar codes to disputes between states lacking formal cooperative arrangements. Post-1975 events were marginally more conflictual than pre-1976 events, with a mean BAR difference of -0.38 (Fig. 4).

### Spatial generalization across continents

Continent-level leave-one-group-out cross-validation provides a spatial complement to the temporal evaluation (Table 5). The model performs best when holding out Europe (QWK 0.417, 959 events) and worst for North America (QWK 0.062, 183 events) and South America (QWK 0.075, 352 events). The mean spatial CV QWK of 0.248 is comparable to the temporal test QWK (0.298), but the range (0.062 to 0.417) reveals that model performance is highly region-dependent. North America's poor spatial CV performance is consistent with its anomalous conflict ratio (37.7%, driven by institutionalised US-Mexico/Canada disagreements coded at BAR intensities comparable to less-institutionalised conflicts elsewhere). These results indicate that the model partially captures region-specific patterns that do not transfer across continents, and that any operational deployment would require region-specific calibration.

**Table 5. Continent-level spatial cross-validation.** Leave-one-continent-out with default XGBoost on 42 non-autoregressive features.

| Holdout Continent | n test | QWK | Macro-F1 |
|:---|:---:|:---:|:---:|
| Africa | 878 | 0.301 | 0.400 |
| Asia | 2,838 | 0.256 | 0.352 |
| Europe | 959 | 0.417 | 0.488 |
| North America | 183 | 0.062 | 0.260 |
| South America | 352 | 0.075 | 0.275 |
| Mean | -- | 0.248 | 0.371 |

### Robustness and sensitivity analyses

Several additional analyses address specific methodological concerns. First, an exhaustive all-subsets analysis of the 7 feature groups (64 subsets containing the baseline) identified baseline + governance + asymmetry + temporal (QWK 0.441) as the best-performing subset, different from all three sequential ablation orderings, confirming that forward selection is a noisy estimator of feature group value. Second, SHAP importance rankings are highly stable between validation and test sets (Spearman rho = 0.984), indicating that while absolute model performance degrades, the relative feature importance structure is preserved. Third, grouped permutation importance for climate features (permuting all five simultaneously) yielded a positive delta of +0.019 on validation QWK using the full 82-feature model, suggesting that climate variables do contribute collectively even when individual features rank low, further supporting the interpretation that the climate null finding is resolution-specific rather than substantive. Fourth, removing treaty formation rate from the 45-feature model improved test QWK from 0.132 to 0.161 (default XGBoost), and removing year improved it to 0.144, consistent with both features encoding temporal dynamics that do not transfer across periods. Fifth, ordinal regression baselines performed better on the full 82-feature set (LogisticAT val QWK 0.242, OrdinalRidge 0.299) than on the pruned 45-feature set (0.174, 0.285), indicating that the LightGBM-optimized feature selection systematically disadvantaged linear models and the full feature set provides a fairer basis for comparison. Sixth, an imputation comparison showed that XGBoost native NaN handling (+0.020 test QWK) and missingness indicators (+0.022) both outperformed median imputation, suggesting that the 60-64% missingness in governance features masks usable signal; re-running the full ablation protocol with native NaN handling is a priority for future work, as governance features may have been erroneously discarded. Finally, treaty formation rate and cooperation momentum exhibit moderate collinearity (Pearson r = 0.303), meaning their SHAP importance estimates share explained variance.

## Discussion

We structure the discussion around three methodological lessons for conflict prediction, followed by tentative policy-relevant hypotheses. An important epistemological caveat applies to the latter: SHAP values measure marginal feature contributions to model predictions, not causal effects on real-world outcomes (Lundberg & Lee, 2017). Features with high SHAP importance may be proxies for true causal factors, reflect confounding, or capture spurious correlations. Without causal identification strategies (directed acyclic graphs, instrumental variables, or natural experiments), all policy-relevant interpretations below are associational hypotheses warranting investigation through causal research designs. We use language such as "consistent with" and "suggests" to reflect this distinction.

### Methodological lesson 1: Autoregressive features are a pervasive trap in conflict prediction

The most consequential methodological finding is that autoregressive features (cooperation momentum, prior event counts, event escalation) inflate validation metrics by nearly 0.10 QWK without improving, and in fact slightly degrading, test-set performance. This is not merely an endogeneity concern in the econometric sense; it is a form of target leakage through a temporally lagged proxy. Cooperation momentum encodes the label distribution of the recent past, which is informative during validation (where the distributional shift from training is small) but misleading during testing (where the shift is larger). Critically, Optuna hyperparameter search exploits this signal aggressively, amplifying the inflation. This finding has broad relevance: any conflict prediction model that includes rolling averages of prior conflict intensity as features is vulnerable to the same trap, and researchers should routinely report performance with and without such features.

### Methodological lesson 2: Sequential ablation is an unreliable estimator of feature importance

The demonstration that three orderings of the same seven feature groups produce three qualitatively different retained sets (Table 2) is a concrete warning to the environmental ML community, which frequently uses forward or backward selection to rank feature importance. The all-subsets analysis (64 combinations, Table in Robustness) identified a best-performing subset (baseline + governance + asymmetry + temporal, QWK 0.441) that differs from all three sequential results. This confirms that sequential ablation conflates marginal and conditional importance: correlated feature groups compete for the same variance, and whichever enters first "wins." Researchers should prefer order-independent methods (grouped permutation importance, Shapley-value decomposition at the group level) or exhaustive enumeration when the number of groups is small.

### Methodological lesson 3: Temporal and spatial generalization set the ceiling for conflict prediction

The validation-to-test gap (26% for the primary model) and the spatial CV variation (QWK 0.062 to 0.417 across continents) are not limitations to be listed and forgotten; they define the practical ceiling of this class of models. Rolling-window analysis (QWK 0.145 to 0.280 across windows) suggests the validation period (1996-2002) may represent an unusually predictable epoch. The post-2003 test period, shaped by the Iraq War and subsequent Middle Eastern geopolitical realignment, introduces distributional shifts that the model cannot accommodate. For operational early warning systems, this implies a requirement for continuous retraining, region-specific calibration, and explicitly communicated uncertainty bounds.

### Policy hypothesis 1: Institutional velocity may matter more than institutional stock

The most policy-relevant finding is the temporal shift in feature importance: treaty formation rate increased 56.5% in SHAP importance from the Cold War to the post-2000 era (with 95% bootstrap CIs reported in Extended Data), while cumulative treaty stock decreased 19.5%. This pattern is consistent with Wolf et al.'s (2003) institutional-change hypothesis, which posits that the rate of institutional adaptation, not the mere existence of agreements, determines whether basins transition toward conflict or cooperation. If this association reflects a genuine mechanism, it carries a specific policy implication: investment in institutional velocity, the capacity to generate new agreements rapidly in response to emerging stresses, may be more effective than investment in comprehensive but static treaty architectures.

This interpretation resonates with Bernauer and Bohmelt's (2020) finding that adaptive management provisions predict treaty durability independently of treaty stock, and with Conca, Wu & Mei's (2006) argument that participatory treaty-making processes generate more flexible frameworks. In practical terms, this suggests that provisional data-sharing agreements, flexible allocation frameworks, and rapid-response institutional mechanisms (such as joint monitoring commissions with adaptive mandate) deserve priority alongside permanent treaty instruments.

However, three caveats temper this conclusion. First, treaty formation rate and cooperation momentum are moderately collinear (Pearson r = 0.303), meaning their SHAP estimates share explained variance. Second, treaties in the TFDD are themselves coded as cooperative events, creating a structural coupling between predictor and target that temporal ordering alone cannot resolve. Third, removing treaty formation rate actually improves test QWK (from 0.132 to 0.161 with default parameters), consistent with the feature encoding temporal dynamics that do not generalize across geopolitical regimes. Causal research designs, such as instrumental variable approaches using exogenous shocks to institutional capacity, are needed to distinguish genuine institutional effects from shared trajectories.

### Policy hypothesis 2: Economic capacity as a governance prerequisite

Economic indicators (GDP per capita, military expenditure, population) are consistently associated with conflict outcomes across ablation orderings, SHAP rankings, and permutation importance. This finding is consistent with the hydro-hegemony framework of Zeitoun and Warner (2006), but carries a policy implication that extends beyond the power-asymmetry framing: economic capacity may function as a prerequisite for effective water governance rather than merely a determinant of bargaining outcomes. States with higher GDP have greater administrative capacity to negotiate, implement, and monitor transboundary agreements, more resources to invest in water infrastructure that reduces zero-sum competition, and stronger institutions that can absorb hydrological shocks without escalation.

This interpretation implies that development assistance and economic capacity-building in water-stressed regions serve a dual function: they address water scarcity directly and strengthen the institutional foundations upon which cooperative outcomes depend. Cascao's (2009) analysis of counter-hegemonic strategies in the Nile basin demonstrates that economic asymmetry is not deterministic; weaker riparian states can resist through coalition-building and norm entrepreneurship. But the consistent predictive power of economic variables across model specifications suggests that material capacity establishes the boundary conditions within which institutional creativity operates.

Our features capture only material power asymmetry (GDP, military expenditure, population). The hydro-hegemony framework identifies additional mechanisms, including discursive power, agenda-setting capacity, and the "sanctioned discourse" (Zeitoun & Warner, 2006), that are not operationalized here. Furthermore, no upstream/downstream position feature is included, despite riparian position being a central determinant of bargaining leverage. The TWINS framework (Mirumachi & Allan, 2007) further implies that a single BAR score per event may capture only one dimension of a multi-dimensional interaction space where conflict and cooperation coexist simultaneously.

### Policy hypothesis 3: Climate as a structural mediator at this resolution

This finding represents a scale mismatch between the climate data specification and the level of analysis, not a refutation of climate's role in water conflict. Two distinct interpretations must be separated: (a) a resolution claim, that basin-averaged annual climate indices are too spatially and temporally aggregated to capture proximate triggers of diplomatic events, and (b) a mechanism claim, that climate operates as a structural condition mediating vulnerability rather than directly triggering events. Our analysis can speak to claim (a) but not to claim (b), which requires causal identification strategies beyond the scope of this correlational exercise. The delta QWK of -0.042 under the primary ordering is consistent with the meta-analytic finding of Mach et al. (2019), who concluded that climate contributes 3-20% of conflict risk with socioeconomic and political factors dominating, and it aligns with the scale-of-analysis argument developed by Ge et al. (2022), who found climate sensitivity at the basin-year level but not necessarily at finer event resolution. Notably, climate variables were retained under the reversed ablation ordering (Table 2), and grouped permutation importance (permuting all five climate features simultaneously) yielded a positive contribution of +0.019 to validation QWK in the full 82-feature model, suggesting that climate features contain collective signal that individual-feature importance metrics miss. The climate null finding is therefore specific to the sequential ablation context and basin-averaged annual specification, not a robust conclusion about climate's predictive irrelevance.

It is essential to be precise about what this finding does and does not establish. The negative result applies specifically to basin-averaged annual climate indices (CRU TS 4.09 precipitation, potential evapotranspiration, SPEI-3) when predicting discrete BAR-coded events. At this aggregation level, climate signals are spatially and temporally diffuse relative to the proximate triggers of diplomatic events. Annual basin-averaged precipitation, computed at 0.5-degree gridded resolution, cannot capture the localised, short-duration hydrological shocks (upstream flow reversals, single-season drought in a critical reach, glacial lake outburst floods) that precipitate crises. This represents a scale mismatch between the climate data specification and the level of analysis, not a refutation of climate's role in water conflict. Higher-resolution climate indicators, such as daily sub-basin precipitation anomalies or reach-specific flow deficits, might capture proximate triggers that annual basin averages miss.

This finding does not invalidate climate as a structural vulnerability factor. Allan (2001) argued that water scarcity operates as a slow-moving stressor whose effects are mediated through political and economic institutions rather than triggering conflict directly, a framing that our results support at the event level. Gleditsch (2012) and Ide (2015) further developed the argument that climate-conflict pathways are indirect, operating through governance capacity, livelihood disruption, and institutional adaptation. Ide et al. (2020) provided empirical evidence that climate effects on conflict are mediated through governance quality, with well-governed states absorbing climate shocks that destabilise weaker institutional contexts. Ge et al. (2022) and Jiang et al. (2025) demonstrate that climatic conditions shape the resource base over which bargaining occurs, a role that operates over decades and is not readily detectable in event-level prediction. These findings are consistent with the broader pattern documented by Buhaug (2010) and Koubi (2019), who showed that climate-conflict statistical relationships are highly sensitive to model specification, variable definition, and spatial scale. Critically, our ablation design tests climate as an additive feature group. If climate operates through governance, as Ide et al. (2020) argue, then adding climate features after governance-related features are already captured in the model will show no marginal gain because the variance is already absorbed. The sequential ablation protocol is structurally incapable of detecting such interaction effects, and future work should test climate-governance interaction terms directly. For practitioners developing climate-sensitive conflict risk frameworks, annual basin-averaged indices remain valuable for identifying long-run vulnerability; they are simply insufficient as standalone proximate-event predictors at this spatiotemporal resolution.

### Policy hypothesis 4: Governance data gaps as a blind spot

An underappreciated finding is that governance indicators (Polity V, WGI) have 60-64% missingness, concentrated in precisely the regions where governance quality matters most for conflict mediation. Median imputation compressed the variance in these features, likely causing them to be erroneously discarded during ablation. When XGBoost's native missing-value handling was used instead, test QWK improved by +0.020, and adding missingness indicators improved it by +0.022, suggesting that the pattern of missing governance data is itself informative (non-random missingness correlated with state fragility).

This has a concrete policy implication: the international community's capacity to predict and prevent water conflict is constrained by the very governance data gaps that characterize conflict-prone states. Investment in governance indicator coverage for water-stressed regions, particularly in Sub-Saharan Africa, Central Asia, and the Middle East, would simultaneously improve predictive models and provide the monitoring infrastructure that institutional frameworks require. The finding that governance features may have been erroneously discarded due to data quality rather than genuine irrelevance suggests that institutional quality may play a larger role in conflict dynamics than our ablation analysis indicates.

### Limitations

Several limitations warrant acknowledgment. First, the validation-to-test performance gap (26% for the primary 42-feature model; 42% when autoregressive features inflate validation metrics) demonstrates that event-level water conflict prediction does not yet generalise reliably across temporal regimes, limiting operational deployment without periodic retraining. Second, the sequential ablation protocol is path-dependent: different feature group orderings produce different retained sets (Table 2), meaning that specific feature group rankings should be interpreted as suggestive rather than definitive. Third, autoregressive features (cooperation momentum, prior event counts) carry endogeneity risk and inflate validation metrics without improving test generalization (Table 4). Fourth, the TFDD ends at 2008, and the dynamics of transboundary water conflict have likely shifted in the subsequent two decades due to climate change acceleration, new dam construction (particularly in the Mekong and Nile basins), and the rise of non-state actors. Fifth, the sample size of 6,805 events, while the largest available for this domain, constrains the complexity of models that can be reliably trained, particularly for the neutral class (4.0% of events, only 34 test samples). Sixth, events within the same basin are not independent; spatial cross-validation (Table 5) reveals substantial geographic variation in model performance, with the model failing for North America and South America (see Results). Seventh, the 4-class ordinal grouping of the BAR scale, while grounded in the original Yoffe et al. coding framework, involves discretisation choices that affect model performance; sensitivity analyses with 3-class and 5-class groupings (Extended Data Table 4) show that 5-class marginally improves validation QWK but no grouping substantially improves test performance. Eighth, the geographic concentration of conflict (81.3% in the top 10 basins) means that model performance is dominated by a small number of high-activity basins, limiting generalisability to low-activity basins. Ninth, raw year of occurrence ranks among the top SHAP features, encoding secular trends that function as a proxy for all time-varying confounders not explicitly modelled; this prevents temporal extrapolation beyond the training period. Specifically, tree-based models assign all years outside the training range to the same leaf as the maximum training-set year, meaning all post-1996 test events are treated identically with respect to this feature, a concrete mechanism contributing to the validation-to-test gap. Era indicators (cold_war, post-2000) provide a more principled but coarser temporal encoding. Finally, governance features (Polity V, WGI) have 60-64% missingness; median imputation may render these features uninformative, and their ablation failure may reflect data quality rather than genuine irrelevance (Extended Data Table 5). An imputation comparison (median vs XGBoost native NaN handling vs missingness indicators) showed that native NaN handling and missingness indicators both improved test QWK by +0.020 and +0.022 respectively over median imputation, suggesting that the imputation strategy affects model performance and that governance features may contain more signal than the median-imputed ablation indicated.

## Conclusion

This study establishes an ordinal machine learning benchmark for transboundary water conflict prediction and exposes three methodological pitfalls with broad relevance to the conflict prediction community. First, autoregressive features derived from prior basin events inflate validation metrics by 0.10 QWK without improving test generalization, a pervasive trap in temporal conflict prediction that researchers should routinely test for by reporting performance with and without such features. Second, sequential feature ablation is path-dependent: the same seven feature groups yield three different retained sets under three orderings, and an all-subsets analysis identifies a different best-performing combination entirely, cautioning against treating ablation rankings as definitive. Third, event-level water conflict prediction does not generalize reliably across temporal regimes (26% validation-to-test gap) or geographic contexts (spatial CV QWK 0.062 to 0.417), establishing sober baseline expectations and implying that operational early warning systems require continuous retraining and region-specific calibration.

From a policy perspective, exploratory SHAP analysis generates four testable hypotheses requiring causal confirmation: (1) the rate of institutional adaptation may matter more than treaty stock for conflict prevention; (2) economic capacity may function as a prerequisite for effective water governance; (3) annual basin-averaged climate indicators at 0.5-degree resolution are insufficient as standalone event-level predictors, though they may contribute collectively as structural mediators; and (4) governance data gaps in conflict-prone regions may cause systematic underestimation of institutional quality's role. These associational patterns point toward specific causal research designs: instrumental variable approaches using exogenous shocks to institutional capacity, quasi-experimental studies of treaty formation under matched basin conditions, and sub-basin sub-annual climate specifications that may capture proximate hydrological triggers. Future work should also extend the analysis to the post-2008 period using updated TFDD releases and re-run the ablation protocol with native missing-value handling to reassess the governance finding.

---

## Data and Code Availability

All analysis code is available at https://github.com/SynBioExplorer/Predicting-the-outcome-of-water-conflicts. The Transboundary Freshwater Dispute Database is maintained by Oregon State University (https://transboundarywaters.ceoas.oregonstate.edu/). World Bank indicators were accessed via the wbgapi Python package. CRU TS 4.09 data are available from the Climatic Research Unit (https://crudata.uea.ac.uk/cru/data/hrg/). SPEI data are available from https://spei.csic.es/. Polity V data are available from the Center for Systemic Peace (https://www.systemicpeace.org/).

---

## Author Contributions

F.M. conceived the study, designed the methodology, performed the analysis, wrote the manuscript, and created the figures.

## Acknowledgements

The Transboundary Freshwater Dispute Database is maintained by the Program in Water Conflict Management and Transformation at Oregon State University. We thank the database curators for making this resource publicly available.

## Competing Interests

The authors declare no competing interests.

---

## References

Allan, J. A. *The Middle East Water Question: Hydropolitics and the Global Economy*. I.B. Tauris (2001).

Buhaug, H. Climate not to blame for African civil wars. *Proc. Natl Acad. Sci. USA* **107**, 16477-16482 (2010).

Bernauer, T. & Bohmelt, T. International water cooperation and interstate conflict. *Glob. Environ. Change* **65**, 102161 (2020).

Brochmann, M. & Hensel, P. R. Peaceful management of international river claims. *Int. Negot.* **14**, 393-418 (2009).

Cascao, A. E. Counter-hegemony and hydropolitics: the Ethiopian challenge to Egyptian hegemony in the Nile Basin. *Water Policy* **11**, 245-265 (2009).

Conca, K., Wu, F. & Mei, C. Global regime formation or complex institution building? The principled content of international river agreements. *Int. Stud. Q.* **50**, 263-285 (2006).

Dinar, S., Katz, D. & Shmueli, D. Does water scarcity lead to cooperation? The role of institutional capacity and governance. *Polit. Geogr.* **48**, 45-55 (2015).

Ge, Q., Hao, M., Ding, F., Jiang, D., Scheffran, J., Helman, D. & Ide, T. Modelling armed conflict risk under climate change with machine learning and time-series data. *Nat. Commun.* **13**, 2839 (2022).

Giordano, M. & Wolf, A. T. Sharing waters: post-Rio international water management. *Nat. Resour. Forum* **27**, 163-171 (2003).

Guyon, I. & Elisseeff, A. An introduction to variable and feature selection. *J. Mach. Learn. Res.* **3**, 1157-1182 (2003).

Gleditsch, N. P. Whither the weather? Climate change and conflict. *J. Peace Res.* **49**, 3-9 (2012).

Gleick, P. H. Water and conflict: fresh water resources and international security. *Int. Secur.* **18**, 79-112 (1993).

Homer-Dixon, T. F. *Environment, Scarcity, and Violence*. Princeton Univ. Press (1999).

Ide, T. Research methods for exploring the links between climate change and conflict. *WIREs Clim. Change* **6**, 369-383 (2015).

Ide, T., Brzoska, M., Donges, J. F. & Schleussner, C.-F. Multi-method evidence for when and how climate-related disasters contribute to armed conflict risk. *Glob. Environ. Change* **62**, 102063 (2020).

Koubi, V. Climate change and conflict. *Annu. Rev. Polit. Sci.* **22**, 343-360 (2019).

Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. *Adv. Neural Inf. Process. Syst.* **30**, 4765-4774 (2017).

Jiang, L., O'Neill, B. C., Zoraghein, H., Dahlke, H. E. & Caldas, M. M. Water-dependent nations are at higher risk of armed conflict. *Nat. Commun.* **16**, 614 (2025).

Mach, K. J., Kraan, C. M., Adger, W. N., Buhaug, H., Burke, M., Fearon, J. D., Field, C. B., Hendrix, C. S., Maystadt, J.-F., O'Loughlin, J., Roessler, P., Scheffran, J., Schultz, K. A. & von Uexkull, N. Climate as a risk factor for armed conflict. *Nature* **571**, 193-197 (2019).

Mirumachi, N. & Allan, J. A. Revisiting transboundary water governance: power, conflict, cooperation and the political economy. *Proc. CAIWA Int. Conf. Adaptive Integ. Water Manage.* (2007).

Unfried, K., Kis-Katos, K. & Poser, T. Water scarcity and social conflict. *J. Environ. Econ. Manage.* **113**, 102633 (2022).

Warner, J. & Zawahri, N. Hegemony and asymmetry: multiple-chessboard games on transboundary rivers. *Int. Environ. Agreem.* **12**, 215-229 (2012).

Wolf, A. T., Yoffe, S. B. & Giordano, M. International waters: identifying basins at risk. *Water Policy* **5**, 29-60 (2003).

Yoffe, S. B., Wolf, A. T. & Giordano, M. Conflict and cooperation over international freshwater resources: indicators of basins at risk. *J. Am. Water Resour. Assoc.* **39**, 1109-1126 (2003).

Zeitoun, M. & Mirumachi, N. Transboundary water interaction I: reconsidering conflict and cooperation. *Int. Environ. Agreem.* **8**, 297-316 (2008).

Zeitoun, M. & Warner, J. Hydro-hegemony: a framework for analysis of trans-boundary water conflicts. *Water Policy* **8**, 435-460 (2006).

---

## Figures

### Figure 1. Model development and evaluation

![](../figures/fig02a_model_comparison.png){width=90%}

![](../figures/fig02b_confusion_matrix_test.png){width=70%}

![](../figures/fig02d_feature_set_comparison.png){width=80%}

**Figure 1.** (**a**) Validation QWK with 95% bootstrap confidence intervals for all models and majority-class baseline. (**b**) Normalised confusion matrix for the best model (nested-CV-tuned XGBoost) on the held-out test set (2003-2008). (**c**) Test set performance comparison between the ablation-pruned feature model and the full 82-feature model, demonstrating that pruning improves generalisation.

\newpage

### Figure 2. SHAP feature importance analysis

![](../figures/fig03a_shap_importance.png){width=90%}

![](../figures/fig03b_shap_beeswarm.png){width=90%}

**Figure 2.** (**a**) Top 15 features ranked by mean absolute SHAP value across all four ordinal classes. (**b**) SHAP summary plot showing the direction and magnitude of each feature's effect on strong cooperation predictions. Colour encodes normalised feature value (blue = low, red = high).

\newpage

### Figure 3. Temporal SHAP decomposition

![](../figures/fig03d_temporal_shap.png){width=90%}

**Figure 3.** Mean absolute SHAP values for selected features computed separately for the Cold War (pre-1990), post-Cold War (1990-1999), and post-2000 (2000-2008) periods. Treaty formation rate increased 56.5% in importance from Cold War to post-2000; cooperation momentum decreased 26.0%; treaty stock decreased 19.5%.

\newpage

### Figure 4. Geographic distribution

![](../figures/fig04a_basin_map_bar_scale.png){width=90%}

![](../figures/fig04b_conflict_hotspots.png){width=90%}

**Figure 4.** (**a**) Global map of 313 transboundary river basins coloured by mean BAR scale (red = conflict, blue = cooperation). (**b**) Conflict hotspot map showing event density for BAR < 0 events. The top 10 basins account for 81.3% of all conflict events.

---

## Extended Data

**Extended Data Table 1.** Complete list of 45 retained features with descriptions, sources, and missingness rates. Available in supplementary data file `ed_table1_features.csv`. Features span five source categories: TFDD spatial database (24 features, 23-31% missing), TFDD events and treaties (5 features, 0-7% missing), World Bank WDI (10 features, 13-48% missing), and derived temporal features (6 features, 0-8% missing).

**Extended Data Table 2.** Per-class precision, recall, and F1 scores for the nested-CV-tuned XGBoost model.

| Split | Class | Precision | Recall | F1 | Support | Predicted n |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Validation | conflict | 0.518 | 0.500 | 0.509 | 354 | 342 |
| Validation | neutral | 0.088 | 0.500 | 0.150 | 70 | 398 |
| Validation | mild coop | 0.643 | 0.287 | 0.397 | 777 | 347 |
| Validation | strong coop | 0.377 | 0.514 | 0.435 | 313 | 427 |
| Test | conflict | 0.473 | 0.494 | 0.483 | 249 | 260 |
| Test | neutral | 0.020 | 0.206 | 0.037 | 34 | 343 |
| Test | mild coop | 0.734 | 0.221 | 0.339 | 625 | 188 |
| Test | strong coop | 0.192 | 0.423 | 0.264 | 97 | 214 |

The model substantially over-predicts the neutral class (343 predicted vs 34 actual on test), indicating that QWK improvement from tuning is partially achieved by redistributing predictions across classes rather than improving class-specific accuracy. The conflict class maintains moderate recall (49.4%) on test data, the most policy-relevant metric.

**Extended Data Table 3.** Basin-level conflict ratios for basins with >20 recorded events (top 10 shown). Full data in `ed_table3_basin_ratios.csv`.

| Basin | Events | Conflict Ratio | Mean BAR |
|:---|:---:|:---:|:---:|
| Nelson-Saskatchewan | 27 | 0.593 | -0.30 |
| Colorado | 23 | 0.522 | 1.00 |
| Rio Grande (North America) | 31 | 0.484 | 0.81 |
| Indus | 280 | 0.425 | 0.39 |
| Helmand | 26 | 0.423 | 0.92 |
| Tigris-Euphrates/Shatt al Arab | 430 | 0.412 | 0.27 |
| Jordan | 488 | 0.406 | 0.19 |
| Salween | 61 | 0.328 | 0.39 |
| Kura-Araks | 30 | 0.300 | 1.00 |
| Ganges-Brahmaputra-Meghna | 309 | 0.256 | 1.06 |

**Extended Data Table 4.** Sensitivity analysis of ordinal class grouping.

| Grouping | Classes | Val QWK | Test QWK | Val F1 | Test F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 3-class | 3 | 0.382 | 0.085 | 0.483 | 0.327 |
| 4-class | 4 | 0.426 | 0.132 | 0.395 | 0.274 |
| 5-class | 5 | 0.441 | 0.140 | 0.349 | 0.225 |

3-class: conflict (BAR<0), neutral (BAR=0), cooperation (BAR>0). 4-class: splits cooperation into mild (BAR 1-3) and strong (BAR>3). 5-class: splits conflict into severe (BAR<-3) and mild (BAR -3 to -1).

The 5-class grouping marginally improves validation QWK (+0.015 vs 4-class) and test QWK (+0.008) but reduces macro-F1 due to smaller per-class sample sizes. The 3-class grouping achieves higher macro-F1 but lower QWK. No grouping substantially improves test generalization, suggesting that the validation-to-test gap reflects temporal regime change rather than target discretisation artefacts.

**Extended Data Table 5.** Features with >50% missingness. Full per-feature missingness in `ed_table5_missingness.csv`.

| Feature | Missing % | Source |
|:---|:---:|:---|
| WGI indicators (RL, PV, GE, CC) | 64.0-64.1 | Worldwide Governance Indicators |
| Polity difference | 64.4 | Polity V (derived) |
| Water stress difference | 61.6 | WDI (derived) |
| AQUASTAT agricultural withdrawal | 53.6 | FAO AQUASTAT |

All WGI and Polity-derived features exceed 60% missingness, meaning median imputation replaces most values with constants. An imputation comparison showed that XGBoost native NaN handling improved test QWK by +0.020 over median imputation (0.152 vs 0.132), and adding missingness indicators improved it by +0.022 (0.154), suggesting these features contain usable signal that median imputation obscures.

![](../figures/ed_fig1_bar_distribution.png){width=80%}

**Extended Data Figure 1.** Distribution of BAR scores across the full dataset with four-class ordinal grouping boundaries (dashed red lines). The neutral class (BAR = 0) contains only 4.0% of events.

![](../figures/ed_fig2_optuna_convergence.png){width=80%}

**Extended Data Figure 2.** Nested-CV Optuna convergence showing trial QWK versus trial number. The best-so-far line (red) plateaus after approximately 20 trials, suggesting the 100-trial budget was sufficient.

![](../figures/ed_fig3_mcnemar.png){width=70%}

**Extended Data Figure 3.** McNemar's test pairwise p-value matrix for all model pairs on the validation set. Values shown are raw p-values; colour intensity encodes -log10(p).
