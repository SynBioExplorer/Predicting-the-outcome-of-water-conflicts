#!/usr/bin/env python3
"""
Round 2 reviewer analyses (Reviewers 4-6).

Run with:
    conda run -n water-conflict python3 scripts/05_reviewer_round2.py
"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import mord
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

PROJ = Path("/Users/felix/Documents/Predicting-the-outcome-of-water-conflicts")
DATA_PROC = PROJ / "data" / "processed"

FEATURE_GROUPS = {
    "baseline_tfdd": [
        "Area_km2_1", "Area_km2_2", "Dams_Exist_1", "Dams_Exist_2",
        "Dam_Plnd_1", "Dam_Plnd_2", "EstDam24_1", "EstDam24_2",
        "runoff_1", "runoff_2", "withdrawal_1", "withdrawal_2",
        "consumpt_1", "consumpt_2", "HydroPolTe_1", "HydroPolTe_2",
        "InstitVuln_1", "InstitVuln_2", "NumberRipa_1", "NumberRipa_2",
        "Wetlands_k_1", "Wetlands_k_2", "PopDen2022_1", "PopDen2022_2",
        "NUMBER_OF_BASINS", "NUMBER_OF_Countries",
        "bilateral", "Issue_Type1", "treaties_before_event",
    ],
    "+climate": ["pre", "pet", "spei", "pre_anomaly", "pre_ltm"],
    "+governance": [
        "polity2_pol1", "polity2_pol2", "RL.EST_wgi1", "RL.EST_wgi2",
        "PV.EST_wgi1", "PV.EST_wgi2", "GE.EST_wgi1", "GE.EST_wgi2",
        "CC.EST_wgi1", "CC.EST_wgi2",
    ],
    "+economic": [
        "NY.GDP.PCAP.CD_wdi1", "NY.GDP.PCAP.CD_wdi2",
        "MS.MIL.XPND.GD.ZS_wdi1", "MS.MIL.XPND.GD.ZS_wdi2",
        "SP.POP.TOTL_wdi1", "SP.POP.TOTL_wdi2",
        "ER.H2O.FWTL.ZS_wdi1", "ER.H2O.FWTL.ZS_wdi2",
        "ER.H2O.INTR.PC_wdi1", "ER.H2O.INTR.PC_wdi2",
    ],
    "+aquastat": [
        "aq_fwtl_zs_aq1", "aq_fwtl_zs_aq2", "aq_intr_pc_aq1",
        "aq_intr_pc_aq2", "aq_fwag_zs_aq1", "aq_fwag_zs_aq2",
    ],
    "+asymmetry": [
        "pop_ratio", "withdrawal_ratio", "dam_ratio", "instit_vuln_diff",
        "hydropol_max", "gdp_ratio", "polity_diff", "water_stress_diff",
    ],
    "+temporal": [
        "events_prior_5yr", "cooperation_momentum", "cold_war",
        "treaty_rate_5yr", "event_escalation", "year",
    ],
}

RETAINED_45 = []
for g in ["baseline_tfdd", "+economic", "+temporal"]:
    RETAINED_45.extend(FEATURE_GROUPS[g])

AR_FEATURES = ["cooperation_momentum", "events_prior_5yr", "event_escalation"]

# Cached best params from nested-CV Optuna (seed=42, 100 trials)
TUNED_PARAMS = {
    "n_estimators": 212, "max_depth": 8,
    "learning_rate": 0.010023389638193707,
    "subsample": 0.8836430638501277,
    "colsample_bytree": 0.8347474246724079,
    "min_child_weight": 6,
    "reg_alpha": 0.00034796152825675056,
    "reg_lambda": 7.73069226267322,
}


def qwk(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def filt(cols, avail):
    return [c for c in cols if c in avail]


def load_and_split():
    df = pd.read_parquet(DATA_PROC / "events_enriched.parquet")
    if "Issue_Type1" in df.columns:
        df["Issue_Type1"] = df["Issue_Type1"].astype("Int64")
    yr = df["year"].fillna(-1).astype(int)
    return df, df[yr < 1996], df[(yr >= 1996) & (yr <= 2002)], df[yr > 2002]


def prep(df_tr, df_va, df_te, feat_cols):
    X_tr = df_tr[feat_cols].astype(float)
    X_va = df_va[feat_cols].astype(float)
    X_te = df_te[feat_cols].astype(float)
    imp = SimpleImputer(strategy="median")
    imp.fit(X_tr)
    y_tr = df_tr["target"].fillna(-1).astype(int).values
    y_va = df_va["target"].fillna(-1).astype(int).values
    y_te = df_te["target"].fillna(-1).astype(int).values
    return imp.transform(X_tr), imp.transform(X_va), imp.transform(X_te), y_tr, y_va, y_te


def train_xgb(X_tr, y_tr, params):
    sw = compute_sample_weight("balanced", y_tr)
    m = xgb.XGBClassifier(
        num_class=4, objective="multi:softmax",
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
        **params,
    )
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


def sep(c="-", w=80):
    print(c * w)


# ===========================================================================
# 1. Autoregressive ablation on TUNED model
# ===========================================================================
def analysis_1_tuned_ar_ablation(df_tr, df_va, df_te, avail):
    sep("=")
    print("1. Autoregressive ablation on TUNED model (CRITICAL)")
    sep("=")

    rows = []
    for label, feat_list in [
        ("tuned_with_ar", filt(RETAINED_45, avail)),
        ("tuned_without_ar", [f for f in filt(RETAINED_45, avail) if f not in AR_FEATURES]),
    ]:
        X_tr, X_va, X_te, y_tr, y_va, y_te = prep(df_tr, df_va, df_te, feat_list)
        m = train_xgb(X_tr, y_tr, TUNED_PARAMS)
        va_q = qwk(y_va, m.predict(X_va))
        te_q = qwk(y_te, m.predict(X_te))
        va_f = macro_f1(y_va, m.predict(X_va))
        te_f = macro_f1(y_te, m.predict(X_te))
        rows.append({"model": label, "n": len(feat_list),
                      "val_qwk": va_q, "test_qwk": te_q,
                      "val_f1": va_f, "test_f1": te_f})
        print(f"  {label:<25} n={len(feat_list):>2}  val_QWK={va_q:.4f}  test_QWK={te_q:.4f}  val_F1={va_f:.4f}  test_F1={te_f:.4f}")

    delta_val = rows[0]["val_qwk"] - rows[1]["val_qwk"]
    delta_test = rows[0]["test_qwk"] - rows[1]["test_qwk"]
    print(f"\n  Delta (with - without AR): val={delta_val:+.4f}  test={delta_test:+.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA_PROC / "r2_tuned_ar_ablation.csv", index=False)
    return df_out


# ===========================================================================
# 2. All-subsets (127) feature group analysis + Shapley values
# ===========================================================================
def analysis_2_all_subsets(df_tr, df_va, df_te, avail):
    sep("=")
    print("2. All-subsets feature group analysis (127 combinations)")
    sep("=")

    from lightgbm import LGBMClassifier
    LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                       num_leaves=31, class_weight="balanced", verbose=-1, random_state=42)

    group_names = list(FEATURE_GROUPS.keys())
    n_groups = len(group_names)

    results = []
    for r in range(1, n_groups + 1):
        for combo in combinations(range(n_groups), r):
            names = [group_names[i] for i in combo]
            # baseline must always be included
            if "baseline_tfdd" not in names:
                continue
            feat_cols = []
            for n in names:
                feat_cols.extend(filt(FEATURE_GROUPS[n], avail))
            feat_cols = list(dict.fromkeys(feat_cols))

            X_tr = df_tr[feat_cols].astype(float)
            X_va = df_va[feat_cols].astype(float)
            y_tr = df_tr["target"].fillna(-1).astype(int).values
            y_va = df_va["target"].fillna(-1).astype(int).values

            imp = SimpleImputer(strategy="median")
            imp.fit(X_tr)
            X_tr_i = imp.transform(X_tr)
            X_va_i = imp.transform(X_va)

            m = LGBMClassifier(**LGBM_PARAMS)
            m.fit(X_tr_i, y_tr)
            val_q = qwk(y_va, m.predict(X_va_i))

            results.append({
                "groups": "+".join(names),
                "n_groups": len(names),
                "n_features": len(feat_cols),
                "val_qwk": val_q,
            })

    df_out = pd.DataFrame(results).sort_values("val_qwk", ascending=False).reset_index(drop=True)
    df_out.to_csv(DATA_PROC / "r2_all_subsets.csv", index=False)

    print(f"  Total subsets tested: {len(df_out)}")
    print(f"  Top 5:")
    for _, row in df_out.head(5).iterrows():
        print(f"    {row['groups']:<60} n={row['n_features']:>2}  QWK={row['val_qwk']:.4f}")

    # Shapley value decomposition at group level
    # For each non-baseline group, compute marginal contribution averaged over all coalitions
    baseline_only_qwk = df_out[df_out["groups"] == "baseline_tfdd"]["val_qwk"].values[0]
    additive_groups = [g for g in group_names if g != "baseline_tfdd"]

    # Build a lookup from frozenset of group names to QWK
    qwk_lookup = {}
    for _, row in df_out.iterrows():
        key = frozenset(row["groups"].split("+"))
        qwk_lookup[key] = row["val_qwk"]

    shapley = {}
    for target_group in additive_groups:
        marginals = []
        other_groups = [g for g in additive_groups if g != target_group]
        for r in range(len(other_groups) + 1):
            for combo in combinations(other_groups, r):
                without_set = frozenset(["baseline_tfdd"] + list(combo))
                with_set = frozenset(["baseline_tfdd"] + list(combo) + [target_group])

                if without_set in qwk_lookup and with_set in qwk_lookup:
                    marginals.append(qwk_lookup[with_set] - qwk_lookup[without_set])

        shapley[target_group] = np.mean(marginals) if marginals else 0.0

    print(f"\n  Group-level Shapley values (marginal contribution to val QWK):")
    for g, v in sorted(shapley.items(), key=lambda x: -x[1]):
        print(f"    {g:<20} {v:+.4f}")

    shap_df = pd.DataFrame([{"group": g, "shapley_value": v} for g, v in shapley.items()])
    shap_df.to_csv(DATA_PROC / "r2_group_shapley.csv", index=False)

    return df_out, shap_df


# ===========================================================================
# 3. Test-set SHAP analysis
# ===========================================================================
def analysis_3_test_shap(df_tr, df_va, df_te, avail):
    sep("=")
    print("3. Test-set SHAP analysis")
    sep("=")

    feat_cols = filt(RETAINED_45, avail)
    X_tr, X_va, X_te, y_tr, y_va, y_te = prep(df_tr, df_va, df_te, feat_cols)
    m = train_xgb(X_tr, y_tr, TUNED_PARAMS)

    explainer = shap.TreeExplainer(m)
    sv_va = explainer.shap_values(X_va)
    sv_te = explainer.shap_values(X_te)

    def mean_abs(sv):
        if isinstance(sv, list):
            return np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
        elif sv.ndim == 3:
            return np.abs(sv).mean(axis=2).mean(axis=0)
        return np.abs(sv).mean(axis=0)

    imp_va = pd.DataFrame({"feature": feat_cols, "val_shap": mean_abs(sv_va).ravel()})
    imp_te = pd.DataFrame({"feature": feat_cols, "test_shap": mean_abs(sv_te).ravel()})
    merged = imp_va.merge(imp_te, on="feature")
    merged["val_rank"] = merged["val_shap"].rank(ascending=False).astype(int)
    merged["test_rank"] = merged["test_shap"].rank(ascending=False).astype(int)
    merged = merged.sort_values("val_rank")

    merged.to_csv(DATA_PROC / "r2_shap_val_vs_test.csv", index=False)

    print("  Top 10 features: validation vs test SHAP ranking")
    print(f"  {'Feature':<40} {'Val Rank':>8} {'Test Rank':>9} {'Val SHAP':>9} {'Test SHAP':>10}")
    for _, r in merged.head(10).iterrows():
        print(f"  {r['feature']:<40} {r['val_rank']:>8} {r['test_rank']:>9} {r['val_shap']:>9.4f} {r['test_shap']:>10.4f}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, p = spearmanr(merged["val_rank"], merged["test_rank"])
    print(f"\n  Spearman rank correlation (val vs test SHAP): rho={rho:.3f}, p={p:.4f}")

    return merged


# ===========================================================================
# 4. Grouped permutation importance for climate
# ===========================================================================
def analysis_4_climate_grouped_perm(df_tr, df_va, avail):
    sep("=")
    print("4. Grouped permutation importance for climate features")
    sep("=")

    # Use all 82 candidate features (baseline + all groups)
    all_feats = []
    for g in FEATURE_GROUPS.values():
        all_feats.extend(filt(g, avail))
    all_feats = list(dict.fromkeys(all_feats))

    climate_feats = filt(FEATURE_GROUPS["+climate"], avail)
    climate_idx = [all_feats.index(f) for f in climate_feats if f in all_feats]

    X_tr = df_tr[all_feats].astype(float)
    X_va = df_va[all_feats].astype(float)
    y_tr = df_tr["target"].fillna(-1).astype(int).values
    y_va = df_va["target"].fillna(-1).astype(int).values

    imp = SimpleImputer(strategy="median")
    imp.fit(X_tr)
    X_tr_i = imp.transform(X_tr)
    X_va_i = imp.transform(X_va)

    from lightgbm import LGBMClassifier
    m = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                       num_leaves=31, class_weight="balanced", verbose=-1, random_state=42)
    m.fit(X_tr_i, y_tr)

    baseline_qwk = qwk(y_va, m.predict(X_va_i))

    rng = np.random.default_rng(42)
    n_repeats = 50
    deltas = []
    for _ in range(n_repeats):
        X_perm = X_va_i.copy()
        perm_idx = rng.permutation(len(X_perm))
        for ci in climate_idx:
            X_perm[:, ci] = X_perm[perm_idx, ci]
        perm_qwk = qwk(y_va, m.predict(X_perm))
        deltas.append(baseline_qwk - perm_qwk)

    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    print(f"  Climate features: {climate_feats}")
    print(f"  Baseline QWK (all 82 features): {baseline_qwk:.4f}")
    print(f"  Grouped permutation importance: {mean_delta:+.4f} +/- {std_delta:.4f}")
    print(f"  Interpretation: {'Climate group contributes positively' if mean_delta > 0.005 else 'Climate group contribution is negligible or negative'}")

    result = {"climate_features": ";".join(climate_feats),
              "baseline_qwk": baseline_qwk,
              "grouped_perm_importance_mean": mean_delta,
              "grouped_perm_importance_std": std_delta}
    pd.DataFrame([result]).to_csv(DATA_PROC / "r2_climate_grouped_perm.csv", index=False)
    return result


# ===========================================================================
# 6. Feature ablation: treaty_rate_5yr and year
# ===========================================================================
def analysis_6_treaty_year_ablation(df_tr, df_va, df_te, avail):
    sep("=")
    print("6. Treaty formation rate and year feature ablation")
    sep("=")

    feat_full = filt(RETAINED_45, avail)
    variants = {
        "full_45": feat_full,
        "minus_treaty_rate": [f for f in feat_full if f != "treaty_rate_5yr"],
        "minus_year": [f for f in feat_full if f != "year"],
        "minus_both": [f for f in feat_full if f not in ("treaty_rate_5yr", "year")],
    }

    rows = []
    for label, feats in variants.items():
        X_tr, X_va, X_te, y_tr, y_va, y_te = prep(df_tr, df_va, df_te, feats)
        m = train_xgb(X_tr, y_tr, {"n_estimators": 500, "learning_rate": 0.05,
                                     "max_depth": 6, "subsample": 0.8,
                                     "colsample_bytree": 0.8})
        va_q = qwk(y_va, m.predict(X_va))
        te_q = qwk(y_te, m.predict(X_te))
        rows.append({"variant": label, "n": len(feats), "val_qwk": va_q, "test_qwk": te_q})
        print(f"  {label:<25} n={len(feats):>2}  val_QWK={va_q:.4f}  test_QWK={te_q:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA_PROC / "r2_treaty_year_ablation.csv", index=False)
    return df_out


# ===========================================================================
# 7. Ordinal regression on full 82 features
# ===========================================================================
def analysis_7_ordinal_82(df_tr, df_va, avail):
    sep("=")
    print("7. Ordinal regression on full 82 features")
    sep("=")

    all_feats = []
    for g in FEATURE_GROUPS.values():
        all_feats.extend(filt(g, avail))
    all_feats = list(dict.fromkeys(all_feats))

    X_tr = df_tr[all_feats].astype(float)
    X_va = df_va[all_feats].astype(float)
    y_tr = df_tr["target"].fillna(-1).astype(int).values
    y_va = df_va["target"].fillna(-1).astype(int).values

    imp = SimpleImputer(strategy="median")
    imp.fit(X_tr)
    X_tr_i = imp.transform(X_tr)
    X_va_i = imp.transform(X_va)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_i)
    X_va_s = sc.transform(X_va_i)

    rows = []
    for model_name, cls in [("LogisticAT", mord.LogisticAT), ("OrdinalRidge", mord.OrdinalRidge)]:
        best_q = -1
        best_a = None
        for alpha in [0.01, 0.1, 1.0, 5.0, 50.0]:
            try:
                m = cls(alpha=alpha)
                m.fit(X_tr_s, y_tr)
                p = np.clip(m.predict(X_va_s).astype(int), 0, 3)
                q = qwk(y_va, p)
                if q > best_q:
                    best_q = q
                    best_a = alpha
            except Exception:
                pass
        rows.append({"model": model_name, "feature_set": "82_full",
                      "best_alpha": best_a, "val_qwk": best_q})
        print(f"  {model_name} (82 features, alpha={best_a}): val_QWK={best_q:.4f}")

    # Compare with 45-feature results
    feat_45 = filt(RETAINED_45, avail)
    X_tr_45 = df_tr[feat_45].astype(float)
    X_va_45 = df_va[feat_45].astype(float)
    imp45 = SimpleImputer(strategy="median")
    imp45.fit(X_tr_45)
    sc45 = StandardScaler()
    X_tr_45s = sc45.fit_transform(imp45.transform(X_tr_45))
    X_va_45s = sc45.transform(imp45.transform(X_va_45))

    for model_name, cls in [("LogisticAT", mord.LogisticAT), ("OrdinalRidge", mord.OrdinalRidge)]:
        best_q = -1
        best_a = None
        for alpha in [0.01, 0.1, 1.0, 5.0, 50.0]:
            try:
                m = cls(alpha=alpha)
                m.fit(X_tr_45s, y_tr)
                p = np.clip(m.predict(X_va_45s).astype(int), 0, 3)
                q = qwk(y_va, p)
                if q > best_q:
                    best_q = q
                    best_a = alpha
            except Exception:
                pass
        rows.append({"model": model_name, "feature_set": "45_pruned",
                      "best_alpha": best_a, "val_qwk": best_q})
        print(f"  {model_name} (45 features, alpha={best_a}): val_QWK={best_q:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA_PROC / "r2_ordinal_82_vs_45.csv", index=False)
    return df_out


# ===========================================================================
# 8. North America basin disaggregation
# ===========================================================================
def analysis_8_na_disaggregation(df):
    sep("=")
    print("8. North America basin disaggregation")
    sep("=")

    cont_col = "Continent__1" if "Continent__1" in df.columns else None
    if cont_col is None:
        print("  No continent column. Skipping.")
        return None

    na = df[df[cont_col] == "NA"].copy()
    print(f"  North America events: {len(na)}")

    rows = []
    for basin, grp in na.groupby("Basin_Name_1"):
        n = len(grp)
        bar = grp["BAR_Scale"]
        rows.append({
            "basin": basin, "n_events": n,
            "conflict_ratio": round((bar < 0).mean(), 3),
            "mean_bar": round(bar.mean(), 2),
            "min_bar": int(bar.min()),
            "max_bar": int(bar.max()),
        })

    na_df = pd.DataFrame(rows).sort_values("n_events", ascending=False).reset_index(drop=True)
    na_df.to_csv(DATA_PROC / "r2_na_disaggregation.csv", index=False)

    print(na_df.to_string(index=False))
    return na_df


# ===========================================================================
# 9. Treaty rate / cooperation momentum correlation
# ===========================================================================
def analysis_9_correlation(df, avail):
    sep("=")
    print("9. Treaty rate / cooperation momentum correlation")
    sep("=")

    from scipy.stats import pearsonr, spearmanr

    cols = ["treaty_rate_5yr", "cooperation_momentum", "events_prior_5yr",
            "treaties_before_event", "year"]
    cols = [c for c in cols if c in avail]

    sub = df[cols].dropna()
    print(f"  n={len(sub)} complete cases")

    pairs = [
        ("treaty_rate_5yr", "cooperation_momentum"),
        ("treaty_rate_5yr", "events_prior_5yr"),
        ("cooperation_momentum", "events_prior_5yr"),
    ]

    rows = []
    for a, b in pairs:
        if a in sub.columns and b in sub.columns:
            pr, pp = pearsonr(sub[a], sub[b])
            sr, sp = spearmanr(sub[a], sub[b])
            rows.append({"feature_a": a, "feature_b": b,
                          "pearson_r": round(pr, 3), "pearson_p": round(pp, 6),
                          "spearman_rho": round(sr, 3), "spearman_p": round(sp, 6)})
            print(f"  {a} vs {b}: Pearson r={pr:.3f} (p={pp:.2e}), Spearman rho={sr:.3f}")

    pd.DataFrame(rows).to_csv(DATA_PROC / "r2_feature_correlations.csv", index=False)


# ===========================================================================
# Main
# ===========================================================================
def main():
    sep("=", 80)
    print("  ROUND 2 REVIEWER ANALYSES (R4-R6)")
    sep("=", 80)

    df, df_tr, df_va, df_te = load_and_split()
    avail = set(df.columns)
    print(f"Loaded: {len(df):,} events, train={len(df_tr):,}, val={len(df_va):,}, test={len(df_te):,}\n")

    analysis_1_tuned_ar_ablation(df_tr, df_va, df_te, avail)
    print()
    analysis_2_all_subsets(df_tr, df_va, df_te, avail)
    print()
    analysis_3_test_shap(df_tr, df_va, df_te, avail)
    print()
    analysis_4_climate_grouped_perm(df_tr, df_va, avail)
    print()
    analysis_6_treaty_year_ablation(df_tr, df_va, df_te, avail)
    print()
    analysis_7_ordinal_82(df_tr, df_va, avail)
    print()
    analysis_8_na_disaggregation(df)
    print()
    analysis_9_correlation(df, avail)

    sep("=", 80)
    print("  ALL ROUND 2 ANALYSES COMPLETE")
    sep("=", 80)


if __name__ == "__main__":
    main()
