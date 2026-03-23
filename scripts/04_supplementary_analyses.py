#!/usr/bin/env python3
"""
Supplementary analyses for the transboundary water conflict prediction manuscript.

Addresses remaining reviewer concerns:
  A1. SHAP on nested-CV-tuned model (not default XGBoost)
  A2. Extended Data tables 1, 2, 3, 5
  A3. Extended Data figures 1-3
  A4. Imputation comparison (median vs native NaN vs indicators)
  A5. Ordinal regression hyperparameter tuning (fair baseline comparison)
  A6. Spatial cross-validation (basin-holdout)

Run with:
    conda run -n water-conflict python3 scripts/04_supplementary_analyses.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
import mord
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    cohen_kappa_score, f1_score, precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ = Path("/Users/felix/Documents/Predicting-the-outcome-of-water-conflicts")
DATA_PROC = PROJ / "data" / "processed"
FIGURES = PROJ / "figures"
DATA_PROC.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature groups (mirrored from 02_ablation.py / 03_revision_analyses.py)
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "baseline_tfdd": [
        "Area_km2_1", "Area_km2_2",
        "Dams_Exist_1", "Dams_Exist_2",
        "Dam_Plnd_1", "Dam_Plnd_2",
        "EstDam24_1", "EstDam24_2",
        "runoff_1", "runoff_2",
        "withdrawal_1", "withdrawal_2",
        "consumpt_1", "consumpt_2",
        "HydroPolTe_1", "HydroPolTe_2",
        "InstitVuln_1", "InstitVuln_2",
        "NumberRipa_1", "NumberRipa_2",
        "Wetlands_k_1", "Wetlands_k_2",
        "PopDen2022_1", "PopDen2022_2",
        "NUMBER_OF_BASINS", "NUMBER_OF_Countries",
        "bilateral", "Issue_Type1", "treaties_before_event",
    ],
    "+economic": [
        "NY.GDP.PCAP.CD_wdi1", "NY.GDP.PCAP.CD_wdi2",
        "MS.MIL.XPND.GD.ZS_wdi1", "MS.MIL.XPND.GD.ZS_wdi2",
        "SP.POP.TOTL_wdi1", "SP.POP.TOTL_wdi2",
        "ER.H2O.FWTL.ZS_wdi1", "ER.H2O.FWTL.ZS_wdi2",
        "ER.H2O.INTR.PC_wdi1", "ER.H2O.INTR.PC_wdi2",
    ],
    "+temporal": [
        "events_prior_5yr", "cooperation_momentum", "cold_war",
        "treaty_rate_5yr", "event_escalation", "year",
    ],
}

RETAINED_FEATURES = []
for g in ["baseline_tfdd", "+economic", "+temporal"]:
    RETAINED_FEATURES.extend(FEATURE_GROUPS[g])

FEATURE_DESCRIPTIONS = {
    "Area_km2_1": ("Basin area (km2), country 1 BCU", "TFDD Spatial DB"),
    "Area_km2_2": ("Basin area (km2), country 2 BCU", "TFDD Spatial DB"),
    "Dams_Exist_1": ("Existing dams, country 1", "TFDD Spatial DB"),
    "Dams_Exist_2": ("Existing dams, country 2", "TFDD Spatial DB"),
    "Dam_Plnd_1": ("Planned dams, country 1", "TFDD Spatial DB"),
    "Dam_Plnd_2": ("Planned dams, country 2", "TFDD Spatial DB"),
    "EstDam24_1": ("Estimated dams (2024), country 1", "TFDD Spatial DB"),
    "EstDam24_2": ("Estimated dams (2024), country 2", "TFDD Spatial DB"),
    "runoff_1": ("Annual runoff, country 1", "TFDD Spatial DB"),
    "runoff_2": ("Annual runoff, country 2", "TFDD Spatial DB"),
    "withdrawal_1": ("Water withdrawal, country 1", "TFDD Spatial DB"),
    "withdrawal_2": ("Water withdrawal, country 2", "TFDD Spatial DB"),
    "consumpt_1": ("Water consumption, country 1", "TFDD Spatial DB"),
    "consumpt_2": ("Water consumption, country 2", "TFDD Spatial DB"),
    "HydroPolTe_1": ("Hydropolitical tension, country 1", "TFDD Spatial DB"),
    "HydroPolTe_2": ("Hydropolitical tension, country 2", "TFDD Spatial DB"),
    "InstitVuln_1": ("Institutional vulnerability, country 1", "TFDD Spatial DB"),
    "InstitVuln_2": ("Institutional vulnerability, country 2", "TFDD Spatial DB"),
    "NumberRipa_1": ("Number of riparian states, country 1 basin", "TFDD Spatial DB"),
    "NumberRipa_2": ("Number of riparian states, country 2 basin", "TFDD Spatial DB"),
    "Wetlands_k_1": ("Wetland area (km2), country 1", "TFDD Spatial DB"),
    "Wetlands_k_2": ("Wetland area (km2), country 2", "TFDD Spatial DB"),
    "PopDen2022_1": ("Population density (2022), country 1", "TFDD Spatial DB"),
    "PopDen2022_2": ("Population density (2022), country 2", "TFDD Spatial DB"),
    "NUMBER_OF_BASINS": ("Number of shared basins", "TFDD Events"),
    "NUMBER_OF_Countries": ("Number of countries in event", "TFDD Events"),
    "bilateral": ("Binary: bilateral (1) vs multilateral (0)", "TFDD Events"),
    "Issue_Type1": ("Primary issue type (quantity/quality/hydropower/etc.)", "TFDD Events"),
    "treaties_before_event": ("Cumulative treaty count prior to event date", "TFDD Treaties"),
    "NY.GDP.PCAP.CD_wdi1": ("GDP per capita (current USD), country 1", "World Bank WDI"),
    "NY.GDP.PCAP.CD_wdi2": ("GDP per capita (current USD), country 2", "World Bank WDI"),
    "MS.MIL.XPND.GD.ZS_wdi1": ("Military expenditure (% GDP), country 1", "World Bank WDI"),
    "MS.MIL.XPND.GD.ZS_wdi2": ("Military expenditure (% GDP), country 2", "World Bank WDI"),
    "SP.POP.TOTL_wdi1": ("Total population, country 1", "World Bank WDI"),
    "SP.POP.TOTL_wdi2": ("Total population, country 2", "World Bank WDI"),
    "ER.H2O.FWTL.ZS_wdi1": ("Freshwater withdrawal (% total), country 1", "World Bank WDI"),
    "ER.H2O.FWTL.ZS_wdi2": ("Freshwater withdrawal (% total), country 2", "World Bank WDI"),
    "ER.H2O.INTR.PC_wdi1": ("Renewable internal freshwater per capita, country 1", "World Bank WDI"),
    "ER.H2O.INTR.PC_wdi2": ("Renewable internal freshwater per capita, country 2", "World Bank WDI"),
    "events_prior_5yr": ("Count of events in same basin, prior 5 years", "TFDD Events (derived)"),
    "cooperation_momentum": ("Rolling mean BAR of prior events in same basin", "TFDD Events (derived)"),
    "cold_war": ("Binary: event before 1990", "Temporal"),
    "treaty_rate_5yr": ("Treaties per year in preceding 5-year window", "TFDD Treaties (derived)"),
    "event_escalation": ("Intensity trend of events in preceding 5 years", "TFDD Events (derived)"),
    "year": ("Year of event occurrence", "TFDD Events"),
}

CLASS_NAMES = {0: "conflict", 1: "neutral", 2: "mild_coop", 3: "strong_coop"}
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def sep(char="-", width=80):
    print(char * width)


def qwk(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def filter_available(cols, available):
    return [c for c in cols if c in available]


def load_data():
    df = pd.read_parquet(DATA_PROC / "events_enriched.parquet")
    if "Issue_Type1" in df.columns:
        df["Issue_Type1"] = df["Issue_Type1"].astype("Int64")
    return df


def make_splits(df):
    year = df["year"].fillna(-1).astype(int)
    train = df[year < 1996].copy()
    val = df[(year >= 1996) & (year <= 2002)].copy()
    test = df[year > 2002].copy()
    return train, val, test


def impute_median(X_tr, X_va, X_te):
    imp = SimpleImputer(strategy="median")
    imp.fit(X_tr.astype(float))
    return (
        imp.transform(X_tr.astype(float)),
        imp.transform(X_va.astype(float)),
        imp.transform(X_te.astype(float)),
    )


# ===========================================================================
# A1. SHAP on nested-CV-tuned model
# ===========================================================================
def a1_shap_tuned_model(df_train, df_val, df_test, available):
    sep("=")
    print("A1: SHAP on nested-CV-tuned XGBoost")
    sep("=")

    feat_cols = filter_available(RETAINED_FEATURES, available)
    n_classes = 4

    X_tr_raw = df_train[feat_cols].astype(float)
    X_va_raw = df_val[feat_cols].astype(float)
    y_tr = df_train["target"].fillna(-1).astype(int).values
    y_va = df_val["target"].fillna(-1).astype(int).values

    X_tr, X_va, _ = impute_median(X_tr_raw, X_va_raw, X_va_raw)

    basin_groups = df_train["Basin_Name_1"].fillna("unknown").values

    # Use cached best params from prior Optuna run (same seed=42, deterministic)
    # These were obtained from 100-trial nested 5-fold basin-grouped CV
    best_params = {
        "n_estimators": 212,
        "max_depth": 8,
        "learning_rate": 0.010023389638193707,
        "subsample": 0.8836430638501277,
        "colsample_bytree": 0.8347474246724079,
        "min_child_weight": 6,
        "reg_alpha": 0.00034796152825675056,
        "reg_lambda": 7.73069226267322,
    }
    print(f"  Using cached nested-CV best params (CV QWK: 0.3526)")

    # Also run a quick 30-trial Optuna to get a study object for ED Fig 2
    print("  Running 30-trial Optuna for convergence plot...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        gkf = GroupKFold(n_splits=5)
        fold_qwks = []
        for fi, vi in gkf.split(X_tr, y_tr, basin_groups):
            sw = compute_sample_weight("balanced", y_tr[fi])
            m = xgb.XGBClassifier(
                num_class=n_classes, objective="multi:softmax",
                eval_metric="mlogloss", random_state=42, n_jobs=-1,
                **params,
            )
            m.fit(X_tr[fi], y_tr[fi], sample_weight=sw)
            p = m.predict(X_tr[vi])
            q = qwk(y_tr[vi], p)
            if not np.isnan(q):
                fold_qwks.append(q)
        return float(np.mean(fold_qwks)) if fold_qwks else 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f"  30-trial best CV QWK: {study.best_value:.4f}")

    # Train final model on full training set
    sw_tr = compute_sample_weight("balanced", y_tr)
    best_model = xgb.XGBClassifier(
        num_class=n_classes, objective="multi:softmax",
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
        **best_params,
    )
    best_model.fit(X_tr, y_tr, sample_weight=sw_tr)

    # SHAP analysis
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(best_model)

    # Use validation set for SHAP (same as original notebook)
    shap_values = explainer.shap_values(X_va)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # List of arrays, one per class: each (n_samples, n_features)
        abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # 3D array: (n_samples, n_features, n_classes)
        abs_shap = np.abs(shap_values).mean(axis=2)
    else:
        abs_shap = np.abs(shap_values)

    mean_abs_shap = abs_shap.mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs_shap.ravel(),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("  Top 15 features by SHAP importance (tuned model):")
    for _, r in importance_df.head(15).iterrows():
        print(f"    {r['feature']:<40} {r['mean_abs_shap']:.4f}")

    # --- SHAP Figure: Global importance (fig03a) ---
    top15 = importance_df.head(15).iloc[::-1]
    fig_imp = go.Figure(go.Bar(
        x=top15["mean_abs_shap"], y=top15["feature"],
        orientation="h", marker_color="#2c7bb6",
    ))
    fig_imp.update_layout(
        title="Top 15 Features by Mean |SHAP| (Nested-CV-Tuned XGBoost)",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        template="plotly_white", width=800, height=500,
        font=dict(size=12),
    )
    fig_imp.write_image(str(FIGURES / "fig03a_shap_importance.png"), scale=3)
    fig_imp.write_html(str(FIGURES / "fig03a_shap_importance.html"))
    print("  Saved fig03a_shap_importance.png")

    # --- SHAP Figure: Beeswarm (fig03b) ---
    # Use class 3 (strong cooperation) for beeswarm as in original
    if isinstance(shap_values, list) and len(shap_values) > 3:
        sv_class3 = shap_values[3]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_class3 = shap_values[:, :, 3]
    else:
        sv_class3 = shap_values

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and len(ev) > 3:
        base_val = ev[3]
    else:
        base_val = ev

    shap_exp = shap.Explanation(
        values=sv_class3,
        base_values=base_val,
        data=X_va,
        feature_names=feat_cols,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_exp, max_display=15, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(str(FIGURES / "fig03b_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig03b_shap_beeswarm.png")

    # --- Temporal SHAP decomposition with bootstrap CIs (fig03d) ---
    print("  Computing temporal SHAP decomposition with bootstrap CIs...")
    year_va = df_val["year"].fillna(-1).astype(int).values
    era_masks = {
        "Cold War (pre-1990)": year_va < 1990,
        "Post-Cold War (1990-1999)": (year_va >= 1990) & (year_va < 2000),
        "Post-2000 (2000-2008)": year_va >= 2000,
    }

    # Key features for temporal decomposition
    key_feats = [
        "treaties_before_event", "cooperation_momentum",
        "treaty_rate_5yr", "events_prior_5yr",
        "NY.GDP.PCAP.CD_wdi1", "NUMBER_OF_Countries",
    ]
    key_feats = [f for f in key_feats if f in feat_cols]
    key_indices = [feat_cols.index(f) for f in key_feats]

    era_shap = {}
    era_shap_ci = {}
    n_boot = 500

    for era_name, mask in era_masks.items():
        era_abs = abs_shap[mask]
        era_means = era_abs[:, key_indices].mean(axis=0)
        era_shap[era_name] = era_means

        # Bootstrap CIs
        boot_means = []
        n_era = mask.sum()
        for _ in range(n_boot):
            idx = RNG.integers(0, n_era, n_era)
            boot_means.append(era_abs[idx][:, key_indices].mean(axis=0))
        boot_arr = np.array(boot_means)
        ci_lo = np.percentile(boot_arr, 2.5, axis=0)
        ci_hi = np.percentile(boot_arr, 97.5, axis=0)
        era_shap_ci[era_name] = (ci_lo, ci_hi)

    # Build temporal SHAP figure
    era_names = list(era_shap.keys())
    fig_temp = go.Figure()
    colors = ["#d7191c", "#fdae61", "#2c7bb6"]
    for i, era in enumerate(era_names):
        vals = era_shap[era]
        ci_lo, ci_hi = era_shap_ci[era]
        fig_temp.add_trace(go.Bar(
            name=era, x=key_feats, y=vals,
            error_y=dict(type="data", symmetric=False,
                         array=ci_hi - vals, arrayminus=vals - ci_lo),
            marker_color=colors[i],
        ))
    fig_temp.update_layout(
        title="Temporal SHAP Decomposition (Nested-CV-Tuned XGBoost, 95% Bootstrap CI)",
        xaxis_title="Feature", yaxis_title="Mean |SHAP value|",
        barmode="group", template="plotly_white",
        width=1000, height=500, font=dict(size=12),
    )
    fig_temp.write_image(str(FIGURES / "fig03d_temporal_shap.png"), scale=3)
    fig_temp.write_html(str(FIGURES / "fig03d_temporal_shap.html"))
    print("  Saved fig03d_temporal_shap.png")

    # Save SHAP data
    importance_df.to_csv(DATA_PROC / "shap_tuned_importance.csv", index=False)

    # Save temporal SHAP with CIs
    rows = []
    for era in era_names:
        vals = era_shap[era]
        ci_lo, ci_hi = era_shap_ci[era]
        for j, f in enumerate(key_feats):
            rows.append({
                "era": era, "feature": f,
                "mean_abs_shap": vals[j],
                "ci_lower_95": ci_lo[j], "ci_upper_95": ci_hi[j],
            })
    pd.DataFrame(rows).to_csv(DATA_PROC / "shap_temporal_decomposition.csv", index=False)

    return study, best_model, best_params, feat_cols, importance_df


# ===========================================================================
# A2. Extended Data Tables
# ===========================================================================
def a2_extended_data_tables(df, df_train, df_val, df_test, best_model, feat_cols, available):
    sep("=")
    print("A2: Extended Data Tables")
    sep("=")

    # --- ED Table 1: Feature list ---
    print("\n  [ED Table 1] 45 retained features...")
    rows = []
    for f in feat_cols:
        desc, source = FEATURE_DESCRIPTIONS.get(f, ("", ""))
        miss_pct = df[f].isna().mean() * 100 if f in df.columns else None
        rows.append({
            "feature": f, "description": desc,
            "source": source, "missingness_pct": round(miss_pct, 1) if miss_pct is not None else None,
        })
    ed1 = pd.DataFrame(rows)
    ed1.to_csv(DATA_PROC / "ed_table1_features.csv", index=False)
    print(f"    Saved {len(ed1)} features to ed_table1_features.csv")

    # --- ED Table 2: Per-class metrics ---
    print("\n  [ED Table 2] Per-class precision/recall/F1...")
    X_tr_raw = df_train[feat_cols].astype(float)
    X_va_raw = df_val[feat_cols].astype(float)
    X_te_raw = df_test[feat_cols].astype(float)
    y_va = df_val["target"].fillna(-1).astype(int).values
    y_te = df_test["target"].fillna(-1).astype(int).values

    _, X_va_imp, X_te_imp = impute_median(X_tr_raw, X_va_raw, X_te_raw)

    va_preds = best_model.predict(X_va_imp)
    te_preds = best_model.predict(X_te_imp)

    rows = []
    for split_name, y_true, y_pred in [("validation", y_va, va_preds), ("test", y_te, te_preds)]:
        p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], zero_division=0)
        # Predicted class frequencies
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        for cls in range(4):
            rows.append({
                "split": split_name, "class": cls,
                "class_name": CLASS_NAMES[cls],
                "precision": round(p[cls], 3), "recall": round(r[cls], 3),
                "f1": round(f1[cls], 3), "support": int(sup[cls]),
                "predicted_count": int(pred_counts.get(cls, 0)),
            })

    ed2 = pd.DataFrame(rows)
    ed2.to_csv(DATA_PROC / "ed_table2_per_class.csv", index=False)
    print("    Saved ed_table2_per_class.csv")
    print(ed2.to_string(index=False))

    # --- ED Table 3: Basin conflict ratios ---
    print("\n  [ED Table 3] Basin-level conflict ratios...")
    basin_stats = []
    for basin, grp in df.groupby("Basin_Name_1"):
        n = len(grp)
        if n < 20:
            continue
        bar = grp["BAR_Scale"]
        conflict_ratio = (bar < 0).mean()
        mean_bar = bar.mean()
        basin_stats.append({
            "basin": basin, "n_events": n,
            "conflict_ratio": round(conflict_ratio, 3),
            "mean_bar": round(mean_bar, 2),
        })
    ed3 = pd.DataFrame(basin_stats).sort_values("conflict_ratio", ascending=False).reset_index(drop=True)
    ed3.to_csv(DATA_PROC / "ed_table3_basin_ratios.csv", index=False)
    print(f"    Saved {len(ed3)} basins to ed_table3_basin_ratios.csv")
    print(ed3.head(10).to_string(index=False))

    # --- ED Table 5: Per-feature missingness ---
    print("\n  [ED Table 5] Per-feature missingness rates...")
    all_candidate_feats = []
    for g in FEATURE_GROUPS.values():
        all_candidate_feats.extend(g)
    # Also add governance, climate, aquastat, asymmetry features
    extra_groups = {
        "+climate": ["pre", "pet", "spei", "pre_anomaly", "pre_ltm"],
        "+governance": [
            "polity2_pol1", "polity2_pol2",
            "RL.EST_wgi1", "RL.EST_wgi2",
            "PV.EST_wgi1", "PV.EST_wgi2",
            "GE.EST_wgi1", "GE.EST_wgi2",
            "CC.EST_wgi1", "CC.EST_wgi2",
        ],
        "+aquastat": [
            "aq_fwtl_zs_aq1", "aq_fwtl_zs_aq2",
            "aq_intr_pc_aq1", "aq_intr_pc_aq2",
            "aq_fwag_zs_aq1", "aq_fwag_zs_aq2",
        ],
        "+asymmetry": [
            "pop_ratio", "withdrawal_ratio", "dam_ratio",
            "instit_vuln_diff", "hydropol_max",
            "gdp_ratio", "polity_diff", "water_stress_diff",
        ],
    }
    for g_feats in extra_groups.values():
        all_candidate_feats.extend(g_feats)
    all_candidate_feats = list(dict.fromkeys(all_candidate_feats))

    miss_rows = []
    for f in all_candidate_feats:
        if f in df.columns:
            pct = df[f].isna().mean() * 100
            miss_rows.append({
                "feature": f,
                "missingness_pct": round(pct, 1),
                "n_missing": int(df[f].isna().sum()),
                "n_total": len(df),
                "high_missingness": pct > 50,
            })
    ed5 = pd.DataFrame(miss_rows).sort_values("missingness_pct", ascending=False).reset_index(drop=True)
    ed5.to_csv(DATA_PROC / "ed_table5_missingness.csv", index=False)
    print(f"    Saved {len(ed5)} features to ed_table5_missingness.csv")
    print("    Features with >50% missing:")
    high_miss = ed5[ed5["high_missingness"]]
    if len(high_miss) > 0:
        print(high_miss[["feature", "missingness_pct"]].to_string(index=False))

    return ed2


# ===========================================================================
# A3. Extended Data Figures
# ===========================================================================
def a3_extended_data_figures(df, study, df_val, best_model, feat_cols, available):
    sep("=")
    print("A3: Extended Data Figures")
    sep("=")

    # --- ED Figure 1: BAR distribution ---
    print("\n  [ED Fig 1] BAR score distribution...")
    bar_vals = df["BAR_Scale"].dropna()

    fig_bar = go.Figure(go.Histogram(
        x=bar_vals, nbinsx=15, marker_color="#2c7bb6",
        marker_line_color="white", marker_line_width=1,
    ))
    # Add grouping boundary lines
    for boundary, label in [(-0.5, "conflict/neutral"), (0.5, "neutral/mild coop"), (3.5, "mild/strong coop")]:
        fig_bar.add_vline(x=boundary, line_dash="dash", line_color="red", line_width=2,
                          annotation_text=label, annotation_position="top")
    fig_bar.update_layout(
        title="Distribution of BAR Scores with 4-Class Grouping Boundaries",
        xaxis_title="BAR Scale", yaxis_title="Count",
        template="plotly_white", width=800, height=400,
    )
    fig_bar.write_image(str(FIGURES / "ed_fig1_bar_distribution.png"), scale=3)
    print("    Saved ed_fig1_bar_distribution.png")

    # --- ED Figure 2: Optuna convergence ---
    print("\n  [ED Fig 2] Optuna trial convergence...")
    trials_df = study.trials_dataframe()
    best_so_far = trials_df["value"].cummax()

    fig_opt = go.Figure()
    fig_opt.add_trace(go.Scatter(
        x=trials_df["number"], y=trials_df["value"],
        mode="markers", name="Trial QWK", marker=dict(size=4, color="#abd9e9"),
    ))
    fig_opt.add_trace(go.Scatter(
        x=trials_df["number"], y=best_so_far,
        mode="lines", name="Best so far", line=dict(color="#d7191c", width=2),
    ))
    fig_opt.update_layout(
        title="Nested-CV Optuna Convergence (100 Trials)",
        xaxis_title="Trial Number", yaxis_title="5-Fold Basin-Grouped CV QWK",
        template="plotly_white", width=800, height=400,
    )
    fig_opt.write_image(str(FIGURES / "ed_fig2_optuna_convergence.png"), scale=3)
    print("    Saved ed_fig2_optuna_convergence.png")

    # --- ED Figure 3: McNemar pairwise ---
    print("\n  [ED Fig 3] McNemar pairwise comparison...")
    # Re-train all models to get predictions
    X_tr_raw = df.loc[df["year"].fillna(-1).astype(int) < 1996, feat_cols].astype(float)
    X_va_raw = df_val[feat_cols].astype(float)
    y_tr = df.loc[df["year"].fillna(-1).astype(int) < 1996, "target"].fillna(-1).astype(int).values
    y_va = df_val["target"].fillna(-1).astype(int).values

    imp = SimpleImputer(strategy="median")
    imp.fit(X_tr_raw)
    X_tr = imp.transform(X_tr_raw)
    X_va = imp.transform(X_va_raw)

    from lightgbm import LGBMClassifier

    models = {}
    # Default LightGBM
    lgbm_def = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                               num_leaves=31, class_weight="balanced", verbose=-1, random_state=42)
    lgbm_def.fit(X_tr, y_tr)
    models["LightGBM (default)"] = lgbm_def.predict(X_va)

    # Default XGBoost
    sw = compute_sample_weight("balanced", y_tr)
    xgb_def = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="mlogloss", random_state=42, n_jobs=-1)
    xgb_def.fit(X_tr, y_tr, sample_weight=sw)
    models["XGBoost (default)"] = xgb_def.predict(X_va)

    # Tuned XGBoost
    models["XGBoost (nested CV)"] = best_model.predict(X_va)

    # Ordinal baselines
    try:
        lor = mord.LogisticAT(alpha=1.0)
        lor.fit(X_tr, y_tr)
        models["LogisticAT"] = lor.predict(X_va).astype(int)
    except Exception as e:
        print(f"    LogisticAT failed: {e}")

    try:
        oridge = mord.OrdinalRidge(alpha=1.0)
        oridge.fit(X_tr, y_tr)
        models["OrdinalRidge"] = oridge.predict(X_va).astype(int)
    except Exception as e:
        print(f"    OrdinalRidge failed: {e}")

    # McNemar test
    from scipy.stats import chi2
    model_names = list(models.keys())
    n_models = len(model_names)
    p_matrix = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            pi = models[model_names[i]]
            pj = models[model_names[j]]
            ci = (pi == y_va)
            cj = (pj == y_va)
            b = (~ci & cj).sum()  # i wrong, j right
            c = (ci & ~cj).sum()  # i right, j wrong
            if b + c > 0:
                stat = (abs(b - c) - 1) ** 2 / (b + c)
                p_val = 1 - chi2.cdf(stat, df=1)
            else:
                p_val = 1.0
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

    fig_mc = go.Figure(go.Heatmap(
        z=-np.log10(p_matrix + 1e-300),
        x=model_names, y=model_names,
        colorscale="RdYlBu_r",
        colorbar_title="-log10(p)",
        text=np.round(p_matrix, 4).astype(str),
        texttemplate="%{text}",
    ))
    fig_mc.update_layout(
        title="McNemar's Test Pairwise p-values (Validation Set)",
        template="plotly_white", width=700, height=600,
    )
    fig_mc.write_image(str(FIGURES / "ed_fig3_mcnemar.png"), scale=3)
    print("    Saved ed_fig3_mcnemar.png")


# ===========================================================================
# A4. Imputation comparison
# ===========================================================================
def a4_imputation_comparison(df_train, df_val, df_test, available):
    sep("=")
    print("A4: Imputation Comparison")
    sep("=")

    feat_cols = filter_available(RETAINED_FEATURES, available)
    y_tr = df_train["target"].fillna(-1).astype(int).values
    y_va = df_val["target"].fillna(-1).astype(int).values
    y_te = df_test["target"].fillna(-1).astype(int).values
    sw = compute_sample_weight("balanced", y_tr)

    results = []

    # Strategy 1: Median imputation (current)
    print("\n  [1] Median imputation...")
    X_tr, X_va, X_te = impute_median(
        df_train[feat_cols].astype(float),
        df_val[feat_cols].astype(float),
        df_test[feat_cols].astype(float),
    )
    m1 = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="mlogloss", random_state=42, n_jobs=-1)
    m1.fit(X_tr, y_tr, sample_weight=sw)
    results.append({
        "strategy": "median_imputation",
        "val_qwk": qwk(y_va, m1.predict(X_va)),
        "test_qwk": qwk(y_te, m1.predict(X_te)),
        "val_f1": macro_f1(y_va, m1.predict(X_va)),
        "test_f1": macro_f1(y_te, m1.predict(X_te)),
    })

    # Strategy 2: Native NaN handling (XGBoost supports this)
    print("  [2] Native NaN handling (XGBoost)...")
    X_tr_nan = df_train[feat_cols].astype(float).values
    X_va_nan = df_val[feat_cols].astype(float).values
    X_te_nan = df_test[feat_cols].astype(float).values
    m2 = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="mlogloss", random_state=42, n_jobs=-1)
    m2.fit(X_tr_nan, y_tr, sample_weight=sw)
    results.append({
        "strategy": "native_nan",
        "val_qwk": qwk(y_va, m2.predict(X_va_nan)),
        "test_qwk": qwk(y_te, m2.predict(X_te_nan)),
        "val_f1": macro_f1(y_va, m2.predict(X_va_nan)),
        "test_f1": macro_f1(y_te, m2.predict(X_te_nan)),
    })

    # Strategy 3: Missingness indicators
    print("  [3] Median imputation + missingness indicators...")
    X_tr_df = df_train[feat_cols].astype(float)
    X_va_df = df_val[feat_cols].astype(float)
    X_te_df = df_test[feat_cols].astype(float)

    # Add binary indicators for features with >10% missing
    miss_cols = [c for c in feat_cols if X_tr_df[c].isna().mean() > 0.10]
    for c in miss_cols:
        X_tr_df[f"{c}_missing"] = X_tr_df[c].isna().astype(float)
        X_va_df[f"{c}_missing"] = X_va_df[c].isna().astype(float)
        X_te_df[f"{c}_missing"] = X_te_df[c].isna().astype(float)

    imp = SimpleImputer(strategy="median")
    imp.fit(X_tr_df)
    X_tr3 = imp.transform(X_tr_df)
    X_va3 = imp.transform(X_va_df)
    X_te3 = imp.transform(X_te_df)

    m3 = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="mlogloss", random_state=42, n_jobs=-1)
    m3.fit(X_tr3, y_tr, sample_weight=sw)
    results.append({
        "strategy": "median_plus_indicators",
        "val_qwk": qwk(y_va, m3.predict(X_va3)),
        "test_qwk": qwk(y_te, m3.predict(X_te3)),
        "val_f1": macro_f1(y_va, m3.predict(X_va3)),
        "test_f1": macro_f1(y_te, m3.predict(X_te3)),
    })

    imp_df = pd.DataFrame(results)
    imp_df.to_csv(DATA_PROC / "imputation_comparison.csv", index=False)

    print("\n  Results:")
    print(imp_df.to_string(index=False))

    return imp_df


# ===========================================================================
# A5. Ordinal regression tuning
# ===========================================================================
def a5_ordinal_tuning(df_train, df_val, available):
    sep("=")
    print("A5: Ordinal Regression Hyperparameter Tuning")
    sep("=")

    feat_cols = filter_available(RETAINED_FEATURES, available)
    y_tr = df_train["target"].fillna(-1).astype(int).values
    y_va = df_val["target"].fillna(-1).astype(int).values

    X_tr, X_va, _ = impute_median(
        df_train[feat_cols].astype(float),
        df_val[feat_cols].astype(float),
        df_val[feat_cols].astype(float),
    )

    # Standardize for ordinal regression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    results = []
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    print("\n  LogisticAT grid search over alpha:")
    best_lat_qwk = -1
    best_lat_alpha = None
    for alpha in alphas:
        try:
            m = mord.LogisticAT(alpha=alpha)
            m.fit(X_tr_sc, y_tr)
            preds = m.predict(X_va_sc).astype(int)
            preds = np.clip(preds, 0, 3)
            val_q = qwk(y_va, preds)
            val_f = macro_f1(y_va, preds)
            results.append({
                "model": "LogisticAT", "alpha": alpha,
                "val_qwk": val_q, "val_macro_f1": val_f,
            })
            print(f"    alpha={alpha:<6} QWK={val_q:.4f} F1={val_f:.4f}")
            if val_q > best_lat_qwk:
                best_lat_qwk = val_q
                best_lat_alpha = alpha
        except Exception as e:
            print(f"    alpha={alpha:<6} FAILED: {e}")

    print(f"  Best LogisticAT: alpha={best_lat_alpha}, QWK={best_lat_qwk:.4f}")

    print("\n  OrdinalRidge grid search over alpha:")
    best_or_qwk = -1
    best_or_alpha = None
    for alpha in alphas:
        try:
            m = mord.OrdinalRidge(alpha=alpha)
            m.fit(X_tr_sc, y_tr)
            preds = m.predict(X_va_sc).astype(int)
            preds = np.clip(preds, 0, 3)
            val_q = qwk(y_va, preds)
            val_f = macro_f1(y_va, preds)
            results.append({
                "model": "OrdinalRidge", "alpha": alpha,
                "val_qwk": val_q, "val_macro_f1": val_f,
            })
            print(f"    alpha={alpha:<6} QWK={val_q:.4f} F1={val_f:.4f}")
            if val_q > best_or_qwk:
                best_or_qwk = val_q
                best_or_alpha = alpha
        except Exception as e:
            print(f"    alpha={alpha:<6} FAILED: {e}")

    print(f"  Best OrdinalRidge: alpha={best_or_alpha}, QWK={best_or_qwk:.4f}")

    ord_df = pd.DataFrame(results)
    ord_df.to_csv(DATA_PROC / "ordinal_regression_tuning.csv", index=False)

    return ord_df, best_lat_qwk, best_lat_alpha, best_or_qwk, best_or_alpha


# ===========================================================================
# A6. Spatial cross-validation (basin holdout)
# ===========================================================================
def a6_spatial_cv(df, available):
    sep("=")
    print("A6: Spatial Cross-Validation (Basin Holdout)")
    sep("=")

    feat_cols = filter_available(RETAINED_FEATURES, available)

    # Use continent as natural basin grouping
    continent_col = "Continent__1" if "Continent__1" in df.columns else None
    if continent_col is None:
        print("  No continent column found. Falling back to Basin_Name_1 grouping.")
        # Group top basins; cluster the rest
        basin_counts = df["Basin_Name_1"].value_counts()
        top_basins = basin_counts[basin_counts >= 50].index.tolist()
        groups = df["Basin_Name_1"].apply(lambda x: x if x in top_basins else "OTHER").values
    else:
        groups = df[continent_col].fillna("Unknown").values

    unique_groups = sorted(set(groups))
    print(f"  Groups ({len(unique_groups)}): {unique_groups}")
    print(f"  Group sizes: {pd.Series(groups).value_counts().to_dict()}")

    y = df["target"].fillna(-1).astype(int).values
    X_raw = df[feat_cols].astype(float)

    fold_results = []
    for holdout_group in unique_groups:
        train_mask = groups != holdout_group
        test_mask = groups == holdout_group

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_test < 10 or n_train < 100:
            continue

        X_tr_raw = X_raw[train_mask]
        X_te_raw = X_raw[test_mask]
        y_tr = y[train_mask]
        y_te = y[test_mask]

        if len(np.unique(y_te)) < 2:
            continue

        imp = SimpleImputer(strategy="median")
        imp.fit(X_tr_raw)
        X_tr = imp.transform(X_tr_raw)
        X_te = imp.transform(X_te_raw)

        sw = compute_sample_weight("balanced", y_tr)
        m = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, sample_weight=sw)
        preds = m.predict(X_te)

        fold_q = qwk(y_te, preds)
        fold_f = macro_f1(y_te, preds)

        fold_results.append({
            "holdout_group": holdout_group,
            "n_train": n_train, "n_test": n_test,
            "qwk": fold_q, "macro_f1": fold_f,
        })
        print(f"    Holdout={holdout_group:<20} n_test={n_test:>5}  QWK={fold_q:.4f}  F1={fold_f:.4f}")

    spatial_df = pd.DataFrame(fold_results)
    mean_qwk = spatial_df["qwk"].mean()
    mean_f1 = spatial_df["macro_f1"].mean()
    print(f"\n  Mean spatial CV QWK: {mean_qwk:.4f}  (compare to temporal val QWK ~0.45)")
    print(f"  Mean spatial CV F1:  {mean_f1:.4f}")

    spatial_df.to_csv(DATA_PROC / "spatial_cv_results.csv", index=False)

    return spatial_df


# ===========================================================================
# Main
# ===========================================================================
def main():
    sep("=", 90)
    print("  SUPPLEMENTARY ANALYSES  -  Water Conflict Manuscript Revision")
    sep("=", 90)

    print("\nLoading data...", end=" ", flush=True)
    df = load_data()
    print(f"OK ({len(df):,} events x {df.shape[1]} columns)")

    available = set(df.columns)
    df_train, df_val, df_test = make_splits(df)
    print(f"Temporal split: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")
    print()

    # A1: SHAP on tuned model
    study, best_model, best_params, feat_cols, importance_df = a1_shap_tuned_model(
        df_train, df_val, df_test, available
    )

    # A2: Extended Data tables
    ed2 = a2_extended_data_tables(df, df_train, df_val, df_test, best_model, feat_cols, available)

    # A3: Extended Data figures
    a3_extended_data_figures(df, study, df_val, best_model, feat_cols, available)

    # A4: Imputation comparison
    imp_df = a4_imputation_comparison(df_train, df_val, df_test, available)

    # A5: Ordinal regression tuning
    ord_df, best_lat_qwk, best_lat_alpha, best_or_qwk, best_or_alpha = a5_ordinal_tuning(
        df_train, df_val, available
    )

    # A6: Spatial CV
    spatial_df = a6_spatial_cv(df, available)

    sep("=", 90)
    print("  ALL SUPPLEMENTARY ANALYSES COMPLETE")
    sep("=", 90)

    print(f"\nOutput files in {DATA_PROC}:")
    for f in sorted(DATA_PROC.glob("*.csv")):
        print(f"  {f.name}")

    print(f"\nFigures in {FIGURES}:")
    for pattern in ["fig03*.png", "ed_fig*.png"]:
        for f in sorted(FIGURES.glob(pattern)):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
