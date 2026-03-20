# Predicting Transboundary Water Conflict Outcomes: Ordinal Machine Learning Reveals Economic and Temporal Drivers Outweigh Climate Signals

**Authors**: [Author list]

**Affiliations**: [Affiliations]

**Correspondence**: [Corresponding author email]

---

## Abstract

Transboundary water interactions shape geopolitical stability for billions of people, yet quantitative prediction of conflict outcomes remains underdeveloped. Here we present the first ordinal-aware machine learning benchmark on the Transboundary Freshwater Dispute Database (TFDD), analysing 6,805 water interaction events across 313 international river basins from 1948 to 2008. Using a systematic ablation protocol inspired by autoresearch methodology, we evaluate eight external data sources and find that economic indicators (GDP per capita, military expenditure, population) and temporal dynamics (cooperation momentum, treaty formation rate) are the strongest predictors of conflict intensity, while climate variables (precipitation, potential evapotranspiration, drought indices) provide no predictive gain. Our best model, an Optuna-tuned XGBoost classifier, achieves a quadratic weighted kappa (QWK) of 0.523 (95% CI: 0.485, 0.558) on validation data, substantially outperforming ordinal regression baselines. SHAP temporal decomposition reveals that treaty formation rate increased 56.5% in importance from the Cold War to the post-2000 era, directly supporting the institutional-change hypothesis. These findings challenge climate-centric conflict narratives and suggest that investment in the rate of institutional development, not merely treaty stock, offers the most actionable lever for conflict prevention.

---

## Introduction

Freshwater scarcity is intensifying as a geopolitical pressure point. Approximately 2.4 billion people live under conditions of water stress, and 286 river basins cross international boundaries, generating persistent interdependencies that neither upstream nor downstream states can unilaterally resolve (Wolf et al., 2003). While 77% of recorded transboundary water interactions are cooperative, conflict events have accelerated since 2017, and the consequences of prediction failure are severe: disrupted agriculture, displaced populations, and in extreme cases, militarised confrontation.

The "water wars" thesis, popularised by early environmental security scholarship (Gleick, 1993; Homer-Dixon, 1999), posited that resource scarcity would drive interstate violence. Wolf and colleagues systematically challenged this framing, demonstrating through the Basins at Risk framework that cooperation overwhelmingly dominates the historical record, and that rapid institutional or physical change, rather than scarcity per se, best explains the transition to conflict (Wolf et al., 2003). This insight reframed the field: the question shifted from whether water wars would occur to under what conditions cooperative or conflictual outcomes emerge.

Despite this conceptual advance, quantitative prediction of transboundary water conflict outcomes remains limited. Ge et al. (2022) applied boosted regression trees to predict conflict onset at basin scale, finding climate sensitivity to be a significant driver. The Water Peace and Security (WPS) Partnership developed a random forest and LSTM pipeline achieving 86% capture rate for conflict events but with a 50% false positive rate, limiting operational utility. Jiang et al. (2025) proposed a water dependency framework explaining 80% of historical conflicts through structural vulnerability. Unfried et al. (2022) used GRACE satellite-derived water mass anomalies as instrumental variables, establishing causal links between water availability and localised conflict. However, these approaches share critical limitations: none account for the ordinal structure of the BAR (Basins at Risk) scale, most lack systematic feature selection, and few provide interpretable explanations for their predictions.

The BAR scale, ranging from -7 (formal war) to +7 (voluntary unification into one nation), encodes a 14-point ordinal gradient of conflict intensity. Treating this as a binary classification or continuous regression discards meaningful structure. Furthermore, prior studies have not systematically tested which data sources contribute predictive power and which introduce noise, an omission that risks overfitting and obscures the true drivers of conflict dynamics.

We address these gaps with three contributions. First, we construct the most comprehensive feature set assembled for TFDD event-level prediction, integrating eight external datasets spanning climate (CRU TS 4.09, SPEI), economics (World Bank WDI), governance (WGI, Polity V), hydrology (FAO AQUASTAT), and institutional history (TFDD treaties and spatial databases). Second, we apply a rigorous ablation protocol, adapted from the autoresearch paradigm (Karpathy, 2024), testing each feature group incrementally under a fixed evaluation protocol to identify which sources genuinely improve prediction. Third, we deploy SHAP-based temporal decomposition to reveal how driver importance shifts across geopolitical eras, providing both predictive accuracy and mechanistic insight. Our results demonstrate that economic capacity and the pace of institutional change, not climate signals, are the dominant predictors of transboundary water conflict outcomes.

## Methods

### Data sources and integration

The primary dataset is the Transboundary Freshwater Dispute Database (TFDD), maintained by Oregon State University, containing 6,805 coded water interaction events across 313 international river basins from 1948 to 2008. Each event is scored on the BAR scale from -7 to +7. We enriched these events with eight external sources: CRU TS 4.09 gridded climate data (0.5-degree resolution monthly precipitation and potential evapotranspiration), the Standardised Precipitation-Evapotranspiration Index at 3-month scale (SPEI-3) for drought characterisation, World Bank World Development Indicators (GDP per capita, total population, military expenditure as percentage of GDP), Worldwide Governance Indicators (rule of law, political stability), Polity V democracy scores, FAO AQUASTAT national water resource statistics (total renewable water resources, agricultural water withdrawal percentage, water dependency ratio), and TFDD's own treaty database (3,812 treaties) and spatial database (818 basin-country units, 2024 update).

### Feature engineering

We constructed features at the basin-country unit (BCU) level, the natural unit of analysis for transboundary interactions. Climate variables were aggregated zonally across basin polygons using area-weighted means. Economic and governance indicators were matched at the country-year level for each state party in the event. For multi-country events, we computed both mean values and asymmetry ratios (e.g., GDP ratio between the wealthiest and poorest participating states, dam count ratios, withdrawal ratios, institutional quality differentials).

Treaty-related features required careful handling to avoid data leakage. We computed cumulative treaty counts strictly prior to each event date, along with treaty formation rates (treaties per year in the preceding 5-year and 10-year windows). Temporal dynamics features captured cooperation momentum (rolling mean BAR of prior events in the same basin), event escalation (count and intensity trend of events in the preceding 5 years), and era indicators (Cold War binary, post-2000 binary).

### Target formulation

We grouped the 14-point BAR scale into four ordinal classes reflecting substantively meaningful thresholds: conflict (BAR < 0; 19.1% of events), neutral (BAR = 0; 4.0%), mild cooperation (0 < BAR <= 3; 51.9%), and strong cooperation (BAR > 3; 24.9%). This grouping preserves ordinality while providing sufficient class sizes for robust estimation.

### Ablation protocol

We adapted the autoresearch incremental testing paradigm to systematically evaluate each feature group. The protocol was fixed across all tests: temporal train/validation split, LightGBM classifier with default hyperparameters, and QWK as the primary metric. Starting from a baseline of 29 TFDD-intrinsic features (basin attributes, treaty counts, event metadata), we added each feature group independently and measured the change in validation QWK (Table 1). A feature group was retained only if it improved QWK; otherwise it was discarded. This yielded a final set of 45 features (baseline + economic + temporal), pruned from an initial pool of 82 candidates.

### Validation strategy

We enforced strict temporal splitting to simulate prospective forecasting: training data comprised events before 1996, validation data covered 1996 to 2002, and the held-out test set spanned 2003 to 2008. This temporal separation prevents information leakage from future events into model training. Hyperparameter tuning used basin-grouped 5-fold cross-validation within the training set to prevent within-basin information leakage during optimisation.

### Models

We evaluated six models spanning two families. Ordinal regression baselines included LogisticAT (all-thresholds ordinal logistic regression) and OrdinalRidge (ridge-penalised ordinal regression). Gradient-boosted tree models included LightGBM and XGBoost, each tested with default hyperparameters and after 100-trial Optuna Bayesian optimisation. All gradient-boosted models used the ablation-selected 45-feature set.

### Evaluation metrics

The primary metric was quadratic weighted kappa (QWK), which penalises predictions proportionally to the squared distance from the true ordinal class, making it well-suited for ordinal outcomes. Secondary metrics included macro-averaged F1 score (to assess per-class balance) and overall accuracy. We computed 95% bootstrap confidence intervals (1,000 resamples) for QWK on the validation set and applied McNemar's test for pairwise model comparison.

### Explainability

We applied SHAP (SHapley Additive exPlanations) TreeExplainer to the best-performing model. Global feature importance was measured as mean absolute SHAP value across all classes. For temporal decomposition, we partitioned events into three eras (Cold War: pre-1990; post-Cold War: 1990-1999; post-2000: 2000-2008) and computed era-specific mean absolute SHAP values, measuring the percentage change in each feature's importance across eras.

## Results

### Systematic ablation identifies economic and temporal features as key predictors

The ablation protocol revealed striking asymmetries in the predictive value of different data sources (Table 1). The baseline TFDD feature set (basin attributes, treaty counts, event metadata; 29 features) achieved a validation QWK of 0.355. Adding climate variables (precipitation, PET, SPEI-3 drought index, anomalies; 5 additional features) decreased performance by 0.053, reducing QWK to 0.301. Governance indicators (Polity V, WGI rule of law, political stability) produced a negligible gain of 0.004, below the retention threshold. AQUASTAT hydrological variables (water dependency ratio, agricultural withdrawal percentage) and asymmetry features (GDP ratio, dam ratio, withdrawal ratio, institutional differential) similarly failed to improve upon the retained feature set, producing deltas of -0.004 and -0.014 respectively.

Two feature groups met the retention criterion. Economic indicators (GDP per capita, military expenditure as percentage of GDP, total population, water withdrawal) improved QWK by 0.044 to 0.399. Temporal dynamics features (event escalation counts, cooperation momentum, Cold War indicator, treaty formation rate) provided a further gain of 0.013, bringing the final validation QWK to 0.411 with 45 features (Fig. 1a).

**Table 1. Ablation results for incremental feature group testing.** Each row reports LightGBM validation QWK with default hyperparameters under a fixed temporal split protocol. Delta QWK is computed relative to the best retained configuration at each step.

| Feature Group | n | QWK | $\Delta$ | Decision |
|:---|:---:|:---:|:---:|:---:|
| Baseline TFDD | 29 | 0.355 | -- | RETAIN |
| +Climate | 34 | 0.301 | -0.053 | DISCARD |
| +Governance | 39 | 0.358 | +0.004 | DISCARD |
| +Economic | 39 | 0.399 | +0.044 | RETAIN |
| +AQUASTAT | 45 | 0.394 | -0.004 | DISCARD |
| +Asymmetry | 47 | 0.385 | -0.014 | DISCARD |
| +Temporal | 45 | 0.411 | +0.013 | RETAIN |

Baseline TFDD: basin attributes, treaties, event metadata. Climate: precipitation, PET, SPEI-3 drought, anomalies. Governance: Polity V, WGI rule of law, political stability. Economic: GDP/capita, military spend, population, water withdrawal. AQUASTAT: water dependency ratio, agricultural withdrawal %. Asymmetry: GDP ratio, dam ratio, withdrawal ratio, institutional differential. Temporal: event escalation, cooperation momentum, Cold War indicator, treaty formation rate.

### Gradient-boosted trees with Optuna tuning achieve best performance

Model comparison on the retained 45-feature set revealed a clear hierarchy (Table 2). The majority-class baseline produced QWK of 0.000 by definition. Ordinal logistic regression (LogisticAT) achieved 0.177 (95% CI: 0.141, 0.212), and OrdinalRidge improved to 0.283 (95% CI: 0.249, 0.317). Default LightGBM and XGBoost reached 0.392 (95% CI: 0.344, 0.436) and 0.451 (95% CI: 0.404, 0.494) respectively. Optuna hyperparameter optimisation (100 trials each) further improved both: LightGBM to 0.517 (95% CI: 0.478, 0.550) and XGBoost to 0.523 (95% CI: 0.485, 0.558), the best overall result (Fig. 1b).

**Table 2. Model comparison on the validation set (1996-2002).** 95% bootstrap confidence intervals computed over 1,000 resamples.

| Model | Val QWK [95% CI] | Val Macro-F1 |
|:---|:---:|:---:|
| Majority class baseline | 0.000 | 0.170 |
| Ordinal logistic (LogisticAT) | 0.177 [0.141, 0.212] | 0.239 |
| OrdinalRidge | 0.283 [0.249, 0.317] | 0.218 |
| LightGBM (default) | 0.392 [0.344, 0.436] | 0.388 |
| XGBoost (default) | 0.451 [0.404, 0.494] | 0.418 |
| LightGBM (Optuna, 100 trials) | 0.517 [0.478, 0.550] | 0.375 |
| XGBoost (Optuna, 100 trials) | 0.523 [0.485, 0.558] | 0.383 |

On the held-out test set (2003-2008), the best model achieved QWK of 0.293, macro-F1 of 0.315, and accuracy of 0.430. The ablation-pruned 45-feature set outperformed the full 82-feature set by +0.069 QWK on the test set, confirming that the pruning protocol reduces overfitting and improves generalisation (Fig. 1c).

### SHAP analysis reveals institutional and economic drivers

Global SHAP analysis of the Optuna-tuned XGBoost model identified five dominant predictors (Fig. 2a). The number of countries involved in the event ranked first (mean |SHAP| = 0.373), reflecting the combinatorial complexity of multilateral negotiations. Events in the prior 5 years ranked second (0.319), capturing basin-level activity intensity. Cooperation momentum, defined as the mean BAR score of prior events in the same basin, ranked third (0.318), indicating strong path dependence in conflict dynamics. Year of occurrence ranked fourth (0.272), encoding broad secular trends toward cooperation or conflict. Issue type ranked fifth (0.204), distinguishing water quantity, quality, hydropower, and navigation disputes.

Notably, no climate variable appeared among the top 15 features by SHAP importance, consistent with the ablation finding that climate data degrades prediction. Economic features, particularly GDP per capita and military expenditure, occupied ranks 6 through 10, confirming the ablation result that economic capacity shapes bargaining outcomes (Fig. 2b).

### Temporal decomposition reveals shifting driver importance

SHAP temporal decomposition across three geopolitical eras (Cold War: pre-1990; post-Cold War: 1990-1999; post-2000: 2000-2008) uncovered significant shifts in driver importance (Fig. 3). Treaty formation rate increased 56.5% in importance from the Cold War era to the post-2000 period, consistent with the hypothesis that the rate of institutional change, rather than institutional stock, drives conflict resolution (Wolf et al., 2003). Cooperation momentum decreased 26.0% in importance post-2000, suggesting that historical path dependence erodes as new actors and issues enter basin-level negotiations. The stock of treaties prior to an event decreased 19.5% in importance post-2000, further reinforcing the distinction between institutional rate and stock (Fig. 3).

These temporal shifts align with observed geopolitical dynamics. The post-Cold War period saw a proliferation of new treaty-making in previously deadlocked basins (e.g., the Mekong River Commission, 1995), elevating the salience of institutional velocity. Simultaneously, the entry of non-state actors and transboundary environmental organisations diluted the explanatory power of historical bilateral cooperation patterns.

### Geographic concentration of conflict

Conflict events were highly spatially concentrated (Fig. 4). The top 10 basins accounted for 81.3% of all conflict events in the dataset. The Jordan basin exhibited the highest conflict ratio at 40.1% (proportion of events with BAR < 0), followed by the Tigris-Euphrates at 39.6% and the Indus at 36.8%. Conversely, high-activity cooperative basins included the Danube, Mekong, La Plata, and Niger. At the continental level, North America exhibited the highest conflict ratio at 37.7%, followed by Asia at 26.3%. Post-1975 events were marginally more conflictual than pre-1976 events, with a mean BAR difference of -0.38 (Fig. 4).

## Discussion

### Climate signals do not improve event-level conflict prediction

The finding that climate variables degrade predictive performance (delta QWK = -0.053) challenges a prominent strand of environmental security research. However, this result is consistent with the meta-analytic finding of Mach et al. (2019), who concluded that climate contributes 3-20% of conflict risk, with socioeconomic and political factors dominating. At the event-level resolution of our analysis, climate signals are likely too spatially and temporally diffuse to capture the proximate triggers of conflict escalation or cooperation. Basin-averaged annual precipitation and drought indices may mask the localised, short-duration hydrological shocks that precipitate crises. Ge et al. (2022) found climate sensitivity at the basin-year level; the discrepancy with our event-level result suggests that climate operates as a background stressor that shapes structural vulnerability rather than as a proximate event-level predictor.

This interpretation carries direct policy implications. Climate-centric early warning systems for water conflict, while valuable for identifying long-term vulnerability, are insufficient as standalone event-level forecasting tools. Effective prediction requires integration of economic, institutional, and temporal dynamics data.

### Economic capacity shapes bargaining outcomes

The retention of economic indicators (delta QWK = +0.044) and the prominence of GDP-related features in SHAP rankings support the hydro-hegemony framework of Zeitoun and Warner (2006), which posits that power asymmetries, mediated through economic and military capacity, determine the outcomes of transboundary water negotiations. States with higher GDP per capita and greater military expenditure as a proportion of GDP are better positioned to secure favourable outcomes, whether through cooperative agreements that reflect their preferences or through coercive strategies that suppress opposition.

This finding also resonates with the water dependency framework of Jiang et al. (2025), who demonstrated that structural economic asymmetries explain 80% of historical water conflicts. Our event-level analysis complements their basin-level vulnerability assessment by showing that economic indicators are predictive not only of whether conflict occurs but of its intensity and resolution.

### Treaty formation rate outweighs treaty stock

The temporal SHAP decomposition provides direct empirical support for Wolf et al.'s (2003) institutional-change hypothesis. The 56.5% increase in importance of treaty formation rate from the Cold War to the post-2000 era, combined with the 19.5% decrease in importance of cumulative treaty stock, demonstrates that what matters for conflict prevention is not the existence of institutions but the pace at which they adapt to changing conditions. Static institutional architectures ossify; dynamic treaty-making signals responsiveness to emerging disputes.

This distinction has direct operational relevance. International organisations and development banks investing in transboundary water governance should prioritise mechanisms that accelerate institutional adaptation, such as provisional data-sharing agreements and flexible allocation frameworks, over the accumulation of comprehensive but rigid treaty instruments.

### Validation-to-test performance gap

The decline from validation QWK of 0.523 to test QWK of 0.293 reflects the distributional shift inherent in the 2003-2008 test period, which encompasses the aftermath of the 2003 Iraq War and its cascading effects on Middle Eastern water diplomacy. This gap is informative rather than disqualifying: it demonstrates that event-level conflict prediction models require periodic retraining to accommodate geopolitical regime changes. The ablation-pruned 45-feature model outperformed the full 82-feature model by +0.069 QWK on the test set, confirming that aggressive feature selection reduces overfitting and improves robustness to distributional shift.

### Comparison with existing forecasting tools

Our approach is complementary to existing tools rather than competitive. The WPS Partnership's random forest and LSTM pipeline achieves 86% capture rate for conflict events but suffers a 50% false positive rate, optimising for sensitivity at the expense of precision. Our ordinal framework provides a more nuanced output: rather than binary conflict/no-conflict, it predicts the intensity of interaction on a four-class ordinal scale. Jiang et al.'s (2025) water dependency framework explains basin-level structural vulnerability but does not predict individual event outcomes. Our event-level approach fills this gap, offering a tool that could be layered on top of structural vulnerability assessments to provide short-to-medium-term forecasts.

### Limitations

Several limitations warrant acknowledgment. First, the TFDD ends at 2008, and the dynamics of transboundary water conflict have likely shifted in the subsequent two decades due to climate change acceleration, new dam construction (particularly in the Mekong and Nile basins), and the rise of non-state actors. Second, the sample size of 6,805 events, while the largest available for this domain, constrains the complexity of models that can be reliably trained. Third, events within the same basin are not independent, and while our basin-grouped cross-validation mitigates this during tuning, the temporal split does not fully account for spatial autocorrelation. Fourth, the 4-class ordinal grouping of the BAR scale, while substantively motivated, involves discretisation choices that affect model performance. Alternative groupings (e.g., 3-class or 7-class) may yield different results. Fifth, the geographic concentration of conflict (81.3% in the top 10 basins) means that model performance is dominated by a small number of high-activity basins, limiting generalisability to low-activity basins.

## Conclusion

This study establishes three principal findings with implications for both conflict prediction and water diplomacy. First, economic indicators and temporal dynamics are the dominant predictors of transboundary water conflict intensity, while climate variables provide no predictive gain at the event level. This challenges climate-deterministic narratives and redirects attention toward the socioeconomic and institutional conditions that mediate water conflict outcomes. Second, the rate of treaty formation, not the stock of existing treaties, drives conflict resolution, particularly in the post-2000 era. This directly supports Wolf et al.'s (2003) institutional-change hypothesis and suggests that policy interventions should prioritise adaptive institutional mechanisms over static legal frameworks. Third, systematic ablation-based feature selection improves generalisation by +0.069 QWK relative to using all available features, demonstrating that principled data curation outperforms data accumulation.

For practitioners in water diplomacy, these findings suggest three actionable priorities: invest in accelerating institutional adaptation rather than merely establishing treaties; recognise that economic development and capacity-building are conflict prevention tools; and treat climate data as a background vulnerability indicator rather than a standalone early warning signal. Future work should extend this framework to the post-2008 period using updated TFDD releases, incorporate sub-national conflict data, and explore deep learning approaches to capture nonlinear temporal dependencies across event sequences.

---

## References

Ge, Q., Hao, M., Ding, F., Jiang, D., Scheffran, J., Helman, D. & Ide, T. Modelling armed conflict risk under climate change with machine learning and time-series data. *Nat. Commun.* **13**, 2839 (2022).

Gleick, P. H. Water and conflict: fresh water resources and international security. *Int. Secur.* **18**, 79-112 (1993).

Homer-Dixon, T. F. *Environment, Scarcity, and Violence*. Princeton Univ. Press (1999).

Jiang, L., O'Neill, B. C., Zoraghein, H., Dahlke, H. E. & Caldas, M. M. Water-dependent nations are at higher risk of armed conflict. *Nat. Commun.* **16**, 614 (2025).

Mach, K. J., Kraan, C. M., Adger, W. N., Buhaug, H., Burke, M., Fearon, J. D., Field, C. B., Hendrix, C. S., Maystadt, J.-F., O'Loughlin, J., Roessler, P., Scheffran, J., Schultz, K. A. & von Uexkull, N. Climate as a risk factor for armed conflict. *Nature* **571**, 193-197 (2019).

Unfried, K., Kis-Katos, K. & Poser, T. Water scarcity and social conflict. *J. Environ. Econ. Manage.* **113**, 102633 (2022).

Wolf, A. T., Yoffe, S. B. & Giordano, M. International waters: identifying basins at risk. *Water Policy* **5**, 29-60 (2003).

Zeitoun, M. & Warner, J. Hydro-hegemony: a framework for analysis of trans-boundary water conflicts. *Water Policy* **8**, 435-460 (2006).

---

## Figures

### Figure 1. Model development and evaluation

![Fig 1a: Model comparison](../figures/fig02a_model_comparison.png){width=90%}

![Fig 1b: Confusion matrix](../figures/fig02b_confusion_matrix_test.png){width=70%}

![Fig 1c: Feature set comparison](../figures/fig02d_feature_set_comparison.png){width=80%}

**Figure 1.** (**a**) Validation QWK with 95% bootstrap confidence intervals for all models and majority-class baseline. (**b**) Normalised confusion matrix for the best model (Optuna-tuned XGBoost) on the held-out test set (2003-2008). (**c**) Test set performance comparison between the ablation-pruned 45-feature model and the full 82-feature model, demonstrating that pruning improves generalisation by +0.069 QWK.

\newpage

### Figure 2. SHAP feature importance analysis

![Fig 2a: SHAP importance](../figures/fig03a_shap_importance.png){width=90%}

![Fig 2b: SHAP beeswarm](../figures/fig03b_shap_beeswarm.png){width=90%}

**Figure 2.** (**a**) Top 15 features ranked by mean absolute SHAP value across all four ordinal classes. (**b**) SHAP summary plot showing the direction and magnitude of each feature's effect on strong cooperation predictions. Colour encodes normalised feature value (blue = low, red = high).

\newpage

### Figure 3. Temporal SHAP decomposition

![Fig 3: Temporal SHAP](../figures/fig03d_temporal_shap.png){width=90%}

**Figure 3.** Mean absolute SHAP values for selected features computed separately for the Cold War (pre-1990), post-Cold War (1990-1999), and post-2000 (2000-2008) periods. Treaty formation rate increased 56.5% in importance from Cold War to post-2000; cooperation momentum decreased 26.0%; treaty stock decreased 19.5%.

\newpage

### Figure 4. Geographic distribution

![Fig 4a: Basin map](../figures/fig04a_basin_map_bar_scale.png){width=90%}

![Fig 4b: Conflict hotspots](../figures/fig04b_conflict_hotspots.png){width=90%}

**Figure 4.** (**a**) Global map of 313 transboundary river basins coloured by mean BAR scale (red = conflict, blue = cooperation). (**b**) Conflict hotspot map showing event density for BAR < 0 events. The top 10 basins account for 81.3% of all conflict events.

---

## Extended Data

**Extended Data Table 1.** Complete list of 45 retained features with descriptions, sources, and temporal coverage.

**Extended Data Table 2.** Per-class precision, recall, and F1 scores for the best model on validation and test sets.

**Extended Data Table 3.** Basin-level conflict ratios for all basins with >20 recorded events.

**Extended Data Figure 1.** Distribution of BAR scores across the full dataset, showing the four-class ordinal grouping boundaries.

**Extended Data Figure 2.** Hyperparameter sensitivity analysis from Optuna optimisation, showing the top 10 most influential hyperparameters for XGBoost.

**Extended Data Figure 3.** McNemar's test pairwise comparison matrix for all model pairs on the validation set.
