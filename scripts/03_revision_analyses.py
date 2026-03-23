#!/usr/bin/env python3
"""
Peer-review revision analyses for the transboundary water conflict prediction project.

Addresses five issues raised by reviewers:
  1. Ablation robustness (path-dependence, bootstrap CIs, permutation importance)
  2. Diagnose val-test QWK gap (concept drift vs. overfitting)
  3. Fix Optuna protocol (nested CV inside training set, not on val set)
  4. Autoregressive feature analysis (circular feature contamination)
  5. Target grouping sensitivity (3-class and 5-class alternatives)

Run with:
    conda run -n water-conflict python3 scripts/03_revision_analyses.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ = Path("/Users/felix/Documents/Predicting-the-outcome-of-water-conflicts")
DATA_PROC = PROJ / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature group definitions (mirrored from 02_ablation.py)
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
    "+climate": ["pre", "pet", "spei", "pre_anomaly", "pre_ltm"],
    "+governance": [
        "polity2_pol1", "polity2_pol2",
        "RL.EST_wgi1", "RL.EST_wgi2",
        "PV.EST_wgi1", "PV.EST_wgi2",
        "GE.EST_wgi1", "GE.EST_wgi2",
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
        "aq_fwtl_zs_aq1", "aq_fwtl_zs_aq2",
        "aq_intr_pc_aq1", "aq_intr_pc_aq2",
        "aq_fwag_zs_aq1", "aq_fwag_zs_aq2",
    ],
    "+asymmetry": [
        "pop_ratio", "withdrawal_ratio", "dam_ratio",
        "instit_vuln_diff", "hydropol_max",
        "gdp_ratio", "polity_diff", "water_stress_diff",
    ],
    "+temporal": [
        "events_prior_5yr", "cooperation_momentum", "cold_war",
        "treaty_rate_5yr", "event_escalation", "year",
    ],
}

# Ablation orderings (group name sequences)
ABLATION_ORDERS = {
    "order_A_original": [
        "baseline_tfdd", "+climate", "+governance", "+economic",
        "+aquastat", "+asymmetry", "+temporal",
    ],
    "order_B_reversed": [
        "baseline_tfdd", "+temporal", "+asymmetry", "+aquastat",
        "+economic", "+governance", "+climate",
    ],
    "order_C_shuffled": [
        "baseline_tfdd", "+economic", "+temporal", "+climate",
        "+governance", "+aquastat", "+asymmetry",
    ],
}

# The 45-feature retained set (baseline + economic + temporal), from ablation
RETAINED_GROUPS = ["baseline_tfdd", "+economic", "+temporal"]
RETAINED_FEATURES = []
for g in RETAINED_GROUPS:
    RETAINED_FEATURES.extend(FEATURE_GROUPS[g])

# Autoregressive (potentially circular) features
AUTOREGRESSIVE_FEATURES = ["cooperation_momentum", "events_prior_5yr", "event_escalation"]

# LightGBM fixed params (same as 02_ablation.py)
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    class_weight="balanced",
    verbose=-1,
    random_state=42,
)

RETAIN_THRESHOLD = 0.005
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sep(char: str = "-", width: int = 80) -> None:
    print(char * width)


def qwk(y_true, y_pred) -> float:
    """Quadratic Weighted Kappa, returns NaN if only one class in y_true."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def filter_available(cols: list, available: set) -> list:
    return [c for c in cols if c in available]


def make_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train.astype(float))
    return imp


def apply_imputer(imp: SimpleImputer, X: pd.DataFrame) -> np.ndarray:
    return imp.transform(X.astype(float))


def get_sample_weights(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight("balanced", y)


def encode_issue_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Issue_Type1" in df.columns:
        df["Issue_Type1"] = df["Issue_Type1"].astype("Int64")
    return df


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROC / "events_enriched.parquet")
    df = encode_issue_type(df)
    return df


def make_splits(df: pd.DataFrame):
    year = df["year"].fillna(-1).astype(int)
    train = df[year < 1996].copy()
    val   = df[(year >= 1996) & (year <= 2002)].copy()
    test  = df[year > 2002].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# XGBoost trainer (used for Issues 3-5 where we tune XGBoost)
# ---------------------------------------------------------------------------

def xgb_default_params() -> dict:
    """Reasonable fixed XGBoost params for non-tuned runs."""
    return dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def train_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    params: dict,
    n_classes: int = 4,
) -> np.ndarray:
    """Train XGBoost, return predictions on X_ev."""
    sw = get_sample_weights(y_tr)
    model = xgb.XGBClassifier(
        num_class=n_classes,
        objective="multi:softmax",
        **params,
    )
    model.fit(X_tr, y_tr, sample_weight=sw)
    return model.predict(X_ev), model


def train_lgbm(X_tr, y_tr, X_ev, feature_names=None):
    """Train LightGBM and predict."""
    model = LGBMClassifier(**LGBM_PARAMS)
    X_tr_f = pd.DataFrame(X_tr, columns=feature_names) if feature_names else X_tr
    X_ev_f = pd.DataFrame(X_ev, columns=feature_names) if feature_names else X_ev
    model.fit(X_tr_f, y_tr)
    return model.predict(X_ev_f), model


# ===========================================================================
# ISSUE 1: Ablation Robustness
# ===========================================================================

def run_single_ablation_order(
    order_name: str,
    group_order: list,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    available: set,
) -> list:
    """Run one ablation ordering; return list of per-group result dicts."""
    retained_cols = []
    best_qwk = -999.0
    rows = []

    for i, group_name in enumerate(group_order):
        group_cols = filter_available(FEATURE_GROUPS[group_name], available)

        if i == 0:
            trial_cols = list(group_cols)
        else:
            trial_cols = list(dict.fromkeys(retained_cols + group_cols))

        # Prepare data
        X_tr = df_train[trial_cols].astype(float)
        X_va = df_val[trial_cols].astype(float)
        y_tr = df_train["target"].fillna(-1).astype(int).values
        y_va = df_val["target"].fillna(-1).astype(int).values

        imp = make_imputer(X_tr)
        X_tr_imp = apply_imputer(imp, X_tr)
        X_va_imp = apply_imputer(imp, X_va)

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr_imp, y_tr)
        val_preds = model.predict(X_va_imp)
        val_qwk = qwk(y_va, val_preds)
        delta = val_qwk - best_qwk

        if i == 0:
            decision = "RETAIN"
            retained_cols = list(trial_cols)
            best_qwk = val_qwk
        elif delta >= RETAIN_THRESHOLD:
            decision = "RETAIN"
            retained_cols = list(trial_cols)
            best_qwk = val_qwk
        else:
            decision = "DISCARD"

        rows.append({
            "order": order_name,
            "step": i,
            "group": group_name,
            "n_features": len(trial_cols),
            "val_qwk": val_qwk,
            "delta_qwk": delta if i > 0 else None,
            "decision": decision,
        })

    return rows


def bootstrap_qwk_delta(
    y_true: np.ndarray,
    y_pred_with: np.ndarray,
    y_pred_without: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Bootstrap CI on QWK delta (with-group minus without-group)."""
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = RNG.integers(0, n, n)
        q_with = qwk(y_true[idx], y_pred_with[idx])
        q_without = qwk(y_true[idx], y_pred_without[idx])
        if not (np.isnan(q_with) or np.isnan(q_without)):
            deltas.append(q_with - q_without)
    deltas = np.array(deltas)
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


def issue1_ablation_robustness(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    available: set,
) -> pd.DataFrame:
    sep("=")
    print("ISSUE 1: Ablation Robustness")
    sep("=")

    all_rows = []

    # 1a. Run 3 orderings
    print("\n[1a] Running 3 ablation orderings ...")
    for order_name, group_order in ABLATION_ORDERS.items():
        print(f"  {order_name} ...", end=" ", flush=True)
        rows = run_single_ablation_order(
            order_name, group_order, df_train, df_val, df_test, available
        )
        all_rows.extend(rows)
        retained = [r["group"] for r in rows if r["decision"] == "RETAIN"]
        print(f"retained groups: {retained}")

    ordering_df = pd.DataFrame(all_rows)

    # 1b. Bootstrap CI on QWK delta for each group (using original order)
    print("\n[1b] Computing bootstrap CIs on QWK delta (1000 resamples) ...")
    # For bootstrap we need val predictions with and without each group
    # Use order A (original) as reference; build cumulative feature sets
    group_order_A = ABLATION_ORDERS["order_A_original"]
    cumulative_cols = []
    bootstrap_rows = []

    y_va = df_val["target"].fillna(-1).astype(int).values

    prev_preds = None
    prev_qwk = None

    for i, group_name in enumerate(group_order_A):
        group_cols = filter_available(FEATURE_GROUPS[group_name], available)
        if i == 0:
            cumulative_cols = list(group_cols)
        else:
            cumulative_cols = list(dict.fromkeys(cumulative_cols + group_cols))

        X_tr = df_train[cumulative_cols].astype(float)
        X_va = df_val[cumulative_cols].astype(float)
        y_tr = df_train["target"].fillna(-1).astype(int).values

        imp = make_imputer(X_tr)
        X_tr_imp = apply_imputer(imp, X_tr)
        X_va_imp = apply_imputer(imp, X_va)

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr_imp, y_tr)
        cur_preds = model.predict(X_va_imp)
        cur_qwk = qwk(y_va, cur_preds)

        if i == 0:
            ci_row = {
                "group": group_name,
                "observed_delta": None,
                "bootstrap_mean_delta": None,
                "ci_lower_95": None,
                "ci_upper_95": None,
                "note": "baseline",
            }
        else:
            obs_delta = cur_qwk - prev_qwk
            ci = bootstrap_qwk_delta(y_va, cur_preds, prev_preds)
            ci_row = {
                "group": group_name,
                "observed_delta": obs_delta,
                "bootstrap_mean_delta": ci["mean_delta"],
                "ci_lower_95": ci["ci_lower"],
                "ci_upper_95": ci["ci_upper"],
                "note": "",
            }
            print(
                f"  {group_name:<18} delta={obs_delta:+.4f}  "
                f"95% CI [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]"
            )

        bootstrap_rows.append(ci_row)
        prev_preds = cur_preds
        prev_qwk = cur_qwk

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # 1c. Permutation importance on final 45-feature model
    print("\n[1c] Permutation importance on final 45-feature model ...")
    feat_cols = filter_available(RETAINED_FEATURES, available)
    X_tr = df_train[feat_cols].astype(float)
    X_va = df_val[feat_cols].astype(float)
    y_tr = df_train["target"].fillna(-1).astype(int).values
    y_va = df_val["target"].fillna(-1).astype(int).values

    imp = make_imputer(X_tr)
    X_tr_imp = apply_imputer(imp, X_tr)
    X_va_imp = apply_imputer(imp, X_va)

    # Sklearn wrapper for qwk scorer
    from sklearn.metrics import make_scorer
    qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")

    lgbm_final = LGBMClassifier(**LGBM_PARAMS)
    lgbm_final.fit(X_tr_imp, y_tr)

    perm_result = permutation_importance(
        lgbm_final, X_va_imp, y_va,
        n_repeats=30,
        random_state=42,
        scoring=qwk_scorer,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": feat_cols,
        "perm_importance_mean": perm_result.importances_mean,
        "perm_importance_std": perm_result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    print("  Top 10 features by permutation importance:")
    for _, r in perm_df.head(10).iterrows():
        print(f"    {r['feature']:<35} {r['perm_importance_mean']:+.4f} +/- {r['perm_importance_std']:.4f}")

    # Combine all results into one output CSV
    # We mark rows by analysis type
    ordering_df["analysis"] = "ordering"
    bootstrap_df["analysis"] = "bootstrap_ci"
    perm_df["analysis"] = "permutation_importance"

    # Save separately with a unified output
    out_df = pd.concat([
        ordering_df.assign(
            observed_delta=ordering_df["delta_qwk"],
            bootstrap_mean_delta=None, ci_lower_95=None, ci_upper_95=None,
            feature=None, perm_importance_mean=None, perm_importance_std=None,
            note=None,
        ),
        bootstrap_df.assign(
            order=None, step=None, n_features=None,
            val_qwk=None, delta_qwk=None, decision=None,
            feature=None, perm_importance_mean=None, perm_importance_std=None,
        ),
        perm_df.assign(
            order=None, step=None, group=None, n_features=None,
            val_qwk=None, delta_qwk=None, decision=None,
            observed_delta=None, bootstrap_mean_delta=None,
            ci_lower_95=None, ci_upper_95=None, note=None,
        ),
    ], ignore_index=True)

    out_path = DATA_PROC / "revision_ablation_robustness.csv"
    out_df.to_csv(out_path, index=False)
    out_df.to_parquet(DATA_PROC / "revision_ablation_robustness.parquet", index=False)
    print(f"\n  Saved to: {out_path}")

    return out_df


# ===========================================================================
# ISSUE 2: Diagnose Val-Test Gap
# ===========================================================================

def issue2_gap_diagnosis(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_full: pd.DataFrame,
    available: set,
) -> pd.DataFrame:
    sep("=")
    print("ISSUE 2: Val-Test QWK Gap Diagnosis")
    sep("=")

    feat_cols = filter_available(RETAINED_FEATURES, available)
    results = []

    # 2a. Class distribution by split
    print("\n[2a] Class distributions by split:")
    for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        counts = df_split["target"].fillna(-1).astype(int).value_counts().sort_index()
        total = len(df_split)
        for cls, cnt in counts.items():
            results.append({
                "analysis": "class_distribution",
                "split": split_name,
                "class": int(cls),
                "count": int(cnt),
                "proportion": cnt / total,
                "metric_name": None,
                "metric_value": None,
            })
        pct = " | ".join(f"cls{c}={cnt/total:.1%}" for c, cnt in counts.items())
        print(f"  {split_name:<6}: n={total:,}  {pct}")

    # 2b. Rolling-window analysis: 15-yr train, 5-yr predict, slide by 5
    print("\n[2b] Rolling-window QWK analysis (15-yr train, 5-yr predict, step=5) ...")
    year_int = df_full["year"].fillna(-1).astype(int)
    all_years = sorted(year_int[year_int > 0].unique())
    min_year, max_year = int(min(all_years)), int(max(all_years))

    WINDOW_TRAIN = 15
    WINDOW_PRED  = 5
    STEP         = 5

    window_start = min_year
    while True:
        train_end   = window_start + WINDOW_TRAIN - 1
        pred_start  = train_end + 1
        pred_end    = pred_start + WINDOW_PRED - 1

        if pred_end > max_year:
            break

        yr = df_full["year"].fillna(-1).astype(int)
        w_train_mask = (yr >= window_start) & (yr <= train_end)
        w_pred_mask  = (yr >= pred_start)  & (yr <= pred_end)

        w_train = df_full[w_train_mask]
        w_pred  = df_full[w_pred_mask]

        if len(w_train) < 50 or len(w_pred) < 20:
            window_start += STEP
            continue

        X_tr = w_train[feat_cols].astype(float)
        X_pr = w_pred[feat_cols].astype(float)
        y_tr = w_train["target"].fillna(-1).astype(int).values
        y_pr = w_pred["target"].fillna(-1).astype(int).values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_pr)) < 2:
            window_start += STEP
            continue

        imp = make_imputer(X_tr)
        X_tr_imp = apply_imputer(imp, X_tr)
        X_pr_imp = apply_imputer(imp, X_pr)

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr_imp, y_tr)
        preds = model.predict(X_pr_imp)
        window_qwk = qwk(y_pr, preds)

        label = f"{window_start}-{train_end} -> {pred_start}-{pred_end}"
        print(f"  Window {label:<30}  n_train={len(w_train):>4}  n_pred={len(w_pred):>4}  QWK={window_qwk:.4f}")
        results.append({
            "analysis": "rolling_window",
            "split": label,
            "class": None,
            "count": len(w_pred),
            "proportion": None,
            "metric_name": "qwk",
            "metric_value": window_qwk,
        })
        window_start += STEP

    # 2c. Train-only -> val and test vs. train+val -> test
    print("\n[2c] Overfitting vs. concept drift diagnosis ...")
    y_va = df_val["target"].fillna(-1).astype(int).values
    y_te = df_test["target"].fillna(-1).astype(int).values

    # Protocol A: train only -> evaluate val and test
    X_tr_A = df_train[feat_cols].astype(float)
    X_va_A = df_val[feat_cols].astype(float)
    X_te_A = df_test[feat_cols].astype(float)
    y_tr_A = df_train["target"].fillna(-1).astype(int).values

    imp_A = make_imputer(X_tr_A)
    X_tr_A_imp = apply_imputer(imp_A, X_tr_A)
    X_va_A_imp = apply_imputer(imp_A, X_va_A)
    X_te_A_imp = apply_imputer(imp_A, X_te_A)

    model_A = LGBMClassifier(**LGBM_PARAMS)
    model_A.fit(X_tr_A_imp, y_tr_A)
    va_preds_A = model_A.predict(X_va_A_imp)
    te_preds_A = model_A.predict(X_te_A_imp)
    qwk_A_val  = qwk(y_va, va_preds_A)
    qwk_A_test = qwk(y_te, te_preds_A)

    print(f"  Protocol A (train -> val,test):  val_QWK={qwk_A_val:.4f}  test_QWK={qwk_A_test:.4f}")

    # Protocol B: train+val -> test
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    X_tv_B = df_trainval[feat_cols].astype(float)
    X_te_B = df_test[feat_cols].astype(float)
    y_tv_B = df_trainval["target"].fillna(-1).astype(int).values

    imp_B = make_imputer(X_tv_B)
    X_tv_B_imp = apply_imputer(imp_B, X_tv_B)
    X_te_B_imp = apply_imputer(imp_B, X_te_B)

    model_B = LGBMClassifier(**LGBM_PARAMS)
    model_B.fit(X_tv_B_imp, y_tv_B)
    te_preds_B = model_B.predict(X_te_B_imp)
    qwk_B_test = qwk(y_te, te_preds_B)

    print(f"  Protocol B (train+val -> test):  test_QWK={qwk_B_test:.4f}")

    gap_A = qwk_A_val - qwk_A_test
    gap_B = qwk_B_test - qwk_A_test
    diagnosis = (
        "CONCEPT_DRIFT" if abs(gap_B) < 0.05
        else "OVERFITTING_LIKELY"
    )
    print(f"  Val-Test gap (Protocol A): {gap_A:+.4f}")
    print(f"  Diagnosis: {diagnosis}")

    for label, mn, mv in [
        ("train_only_to_val",   "qwk", qwk_A_val),
        ("train_only_to_test",  "qwk", qwk_A_test),
        ("train_val_to_test",   "qwk", qwk_B_test),
        ("val_test_gap_A",      "qwk_delta", gap_A),
        ("diagnosis",           "diagnosis_code", float(1 if diagnosis == "CONCEPT_DRIFT" else 0)),
    ]:
        results.append({
            "analysis": "protocol_comparison",
            "split": label,
            "class": None,
            "count": None,
            "proportion": None,
            "metric_name": mn,
            "metric_value": mv,
        })

    out_df = pd.DataFrame(results)
    out_path = DATA_PROC / "revision_gap_diagnosis.csv"
    out_df.to_csv(out_path, index=False)
    out_df.to_parquet(DATA_PROC / "revision_gap_diagnosis.parquet", index=False)
    print(f"\n  Saved to: {out_path}")

    return out_df


# ===========================================================================
# ISSUE 3: Fix Optuna Protocol (Nested CV)
# ===========================================================================

def nested_cv_optuna_objective(
    trial: optuna.Trial,
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    groups: np.ndarray,
    n_classes: int = 4,
) -> float:
    """Optuna objective: 5-fold basin-grouped CV QWK inside training set."""
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

    for fold_train_idx, fold_val_idx in gkf.split(X_train_all, y_train_all, groups):
        X_f_tr = X_train_all[fold_train_idx]
        y_f_tr = y_train_all[fold_train_idx]
        X_f_va = X_train_all[fold_val_idx]
        y_f_va = y_train_all[fold_val_idx]

        sw = get_sample_weights(y_f_tr)
        model = xgb.XGBClassifier(
            num_class=n_classes,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            **params,
        )
        model.fit(X_f_tr, y_f_tr, sample_weight=sw)
        preds = model.predict(X_f_va)
        fold_qwks.append(qwk(y_f_va, preds))

    valid_qwks = [q for q in fold_qwks if not np.isnan(q)]
    return float(np.mean(valid_qwks)) if valid_qwks else 0.0


def issue3_nested_optuna(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    available: set,
) -> pd.DataFrame:
    sep("=")
    print("ISSUE 3: Fix Optuna Protocol (Nested CV inside training set)")
    sep("=")

    feat_cols = filter_available(RETAINED_FEATURES, available)
    n_classes = 4

    X_tr_raw = df_train[feat_cols].astype(float)
    X_va_raw = df_val[feat_cols].astype(float)
    X_te_raw = df_test[feat_cols].astype(float)
    y_tr = df_train["target"].fillna(-1).astype(int).values
    y_va = df_val["target"].fillna(-1).astype(int).values
    y_te = df_test["target"].fillna(-1).astype(int).values

    # Impute
    imp = make_imputer(X_tr_raw)
    X_tr = apply_imputer(imp, X_tr_raw)
    X_va = apply_imputer(imp, X_va_raw)
    X_te = apply_imputer(imp, X_te_raw)

    # Basin groups for GroupKFold (use Basin_Name_1 from training set)
    basin_groups = df_train["Basin_Name_1"].fillna("unknown").values

    print(f"\n  Training set: {len(y_tr):,} events  |  Unique basins: {len(np.unique(basin_groups))}")
    print("  Running 100 Optuna trials with 5-fold basin-grouped CV ...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: nested_cv_optuna_objective(
            trial, X_tr, y_tr, basin_groups, n_classes
        ),
        n_trials=100,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_cv_qwk = study.best_value
    print(f"  Best nested-CV QWK: {best_cv_qwk:.4f}")
    print(f"  Best params: {best_params}")

    # Evaluate best model on val and test
    sw_tr = get_sample_weights(y_tr)
    best_model = xgb.XGBClassifier(
        num_class=n_classes,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        **best_params,
    )
    best_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    va_preds = best_model.predict(X_va)
    te_preds = best_model.predict(X_te)

    nested_val_qwk  = qwk(y_va, va_preds)
    nested_test_qwk = qwk(y_te, te_preds)
    nested_val_f1   = macro_f1(y_va, va_preds)
    nested_test_f1  = macro_f1(y_te, te_preds)

    print(f"\n  [Nested CV tuned XGBoost]  val_QWK={nested_val_qwk:.4f}  test_QWK={nested_test_qwk:.4f}")

    # Baseline: Optuna on validation set (original protocol, replicated here)
    print("\n  [Original Optuna-on-val protocol for comparison] ...")

    def orig_objective(trial):
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
        sw = get_sample_weights(y_tr)
        model = xgb.XGBClassifier(
            num_class=n_classes,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            **params,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)
        preds = model.predict(X_va)
        return qwk(y_va, preds)

    study_orig = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_orig.optimize(orig_objective, n_trials=100, show_progress_bar=False)

    orig_best_params = study_orig.best_params
    orig_val_qwk_tuned = study_orig.best_value

    orig_model = xgb.XGBClassifier(
        num_class=n_classes,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        **orig_best_params,
    )
    orig_model.fit(X_tr, y_tr, sample_weight=get_sample_weights(y_tr))
    orig_te_preds = orig_model.predict(X_te)
    orig_va_preds = orig_model.predict(X_va)
    orig_test_qwk = qwk(y_te, orig_te_preds)
    orig_val_f1   = macro_f1(y_va, orig_va_preds)
    orig_test_f1  = macro_f1(y_te, orig_te_preds)

    print(f"  [Original Optuna-on-val]  val_QWK={orig_val_qwk_tuned:.4f}  test_QWK={orig_test_qwk:.4f}")

    rows = [
        {
            "protocol": "nested_cv_optuna",
            "n_trials": 100,
            "best_cv_qwk": best_cv_qwk,
            "val_qwk": nested_val_qwk,
            "test_qwk": nested_test_qwk,
            "val_macro_f1": nested_val_f1,
            "test_macro_f1": nested_test_f1,
            "note": "5-fold basin-grouped CV inside training set",
        },
        {
            "protocol": "original_optuna_on_val",
            "n_trials": 100,
            "best_cv_qwk": orig_val_qwk_tuned,
            "val_qwk": orig_val_qwk_tuned,
            "test_qwk": orig_test_qwk,
            "val_macro_f1": orig_val_f1,
            "test_macro_f1": orig_test_f1,
            "note": "Original: tunes directly on val set (data leakage)",
        },
    ]

    out_df = pd.DataFrame(rows)
    out_path = DATA_PROC / "revision_nested_optuna.csv"
    out_df.to_csv(out_path, index=False)
    out_df.to_parquet(DATA_PROC / "revision_nested_optuna.parquet", index=False)
    print(f"\n  Saved to: {out_path}")

    return out_df


# ===========================================================================
# ISSUE 4: Autoregressive Feature Analysis
# ===========================================================================

def issue4_autoregressive(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    available: set,
) -> pd.DataFrame:
    sep("=")
    print("ISSUE 4: Autoregressive Feature Analysis")
    sep("=")

    feat_cols_all = filter_available(RETAINED_FEATURES, available)
    feat_cols_no_ar = [c for c in feat_cols_all if c not in AUTOREGRESSIVE_FEATURES]
    ar_present = [c for c in AUTOREGRESSIVE_FEATURES if c in feat_cols_all]

    print(f"\n  Full feature set:          {len(feat_cols_all)} features")
    print(f"  Without autoregressive:    {len(feat_cols_no_ar)} features")
    print(f"  Autoregressive cols found: {ar_present}")

    y_va = df_val["target"].fillna(-1).astype(int).values
    y_te = df_test["target"].fillna(-1).astype(int).values
    n_classes = 4

    rows = []

    for label, feat_cols in [
        ("with_autoregressive", feat_cols_all),
        ("without_autoregressive", feat_cols_no_ar),
    ]:
        X_tr_raw = df_train[feat_cols].astype(float)
        X_va_raw = df_val[feat_cols].astype(float)
        X_te_raw = df_test[feat_cols].astype(float)
        y_tr = df_train["target"].fillna(-1).astype(int).values

        imp = make_imputer(X_tr_raw)
        X_tr = apply_imputer(imp, X_tr_raw)
        X_va = apply_imputer(imp, X_va_raw)
        X_te = apply_imputer(imp, X_te_raw)

        sw = get_sample_weights(y_tr)
        model = xgb.XGBClassifier(
            num_class=n_classes,
            objective="multi:softmax",
            **xgb_default_params(),
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

        va_preds = model.predict(X_va)
        te_preds = model.predict(X_te)

        val_q  = qwk(y_va, va_preds)
        test_q = qwk(y_te, te_preds)
        val_f  = macro_f1(y_va, va_preds)
        test_f = macro_f1(y_te, te_preds)

        print(f"\n  [{label}]")
        print(f"    val_QWK={val_q:.4f}  test_QWK={test_q:.4f}")
        print(f"    val_F1={val_f:.4f}   test_F1={test_f:.4f}")

        rows.append({
            "model": label,
            "n_features": len(feat_cols),
            "autoregressive_removed": label == "without_autoregressive",
            "ar_features_removed": ";".join(ar_present) if label == "without_autoregressive" else "",
            "val_qwk": val_q,
            "test_qwk": test_q,
            "val_macro_f1": val_f,
            "test_macro_f1": test_f,
        })

    # Compute QWK delta
    qwk_delta_val  = rows[0]["val_qwk"]  - rows[1]["val_qwk"]
    qwk_delta_test = rows[0]["test_qwk"] - rows[1]["test_qwk"]
    print(f"\n  QWK delta (with - without AR):  val={qwk_delta_val:+.4f}  test={qwk_delta_test:+.4f}")

    rows.append({
        "model": "ar_contribution_delta",
        "n_features": len(ar_present),
        "autoregressive_removed": None,
        "ar_features_removed": ";".join(ar_present),
        "val_qwk": qwk_delta_val,
        "test_qwk": qwk_delta_test,
        "val_macro_f1": None,
        "test_macro_f1": None,
    })

    out_df = pd.DataFrame(rows)
    out_path = DATA_PROC / "revision_autoregressive.csv"
    out_df.to_csv(out_path, index=False)
    out_df.to_parquet(DATA_PROC / "revision_autoregressive.parquet", index=False)
    print(f"\n  Saved to: {out_path}")

    return out_df


# ===========================================================================
# ISSUE 5: Target Grouping Sensitivity
# ===========================================================================

def make_3class_target(df: pd.DataFrame) -> pd.Series:
    """
    3-class: conflict (BAR<0)=0, neutral (BAR=0)=1, cooperation (BAR>0)=2
    """
    bar = df["BAR_Scale"]
    t = pd.Series(index=df.index, dtype="Int64")
    t[bar < 0]  = 0
    t[bar == 0] = 1
    t[bar > 0]  = 2
    return t


def make_5class_target(df: pd.DataFrame) -> pd.Series:
    """
    5-class:
      strong conflict (BAR<=-3)=0
      mild conflict (-3<BAR<0)=1
      neutral (BAR=0)=2
      mild coop (0<BAR<=3)=3
      strong coop (BAR>3)=4
    """
    bar = df["BAR_Scale"]
    t = pd.Series(index=df.index, dtype="Int64")
    t[bar <= -3]               = 0
    t[(bar > -3) & (bar < 0)] = 1
    t[bar == 0]                = 2
    t[(bar > 0) & (bar <= 3)] = 3
    t[bar > 3]                 = 4
    return t


def issue5_target_sensitivity(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    available: set,
) -> pd.DataFrame:
    sep("=")
    print("ISSUE 5: Target Grouping Sensitivity")
    sep("=")

    feat_cols = filter_available(RETAINED_FEATURES, available)
    rows = []

    groupings = {
        "4class_original": {
            "n_classes": 4,
            "target_fn": lambda df: df["target"].fillna(-1).astype(int),
        },
        "3class": {
            "n_classes": 3,
            "target_fn": make_3class_target,
        },
        "5class": {
            "n_classes": 5,
            "target_fn": make_5class_target,
        },
    }

    for grouping_name, cfg in groupings.items():
        n_cls = cfg["n_classes"]
        print(f"\n  [{grouping_name}] ({n_cls} classes)")

        y_tr = cfg["target_fn"](df_train).values.astype(int)
        y_va = cfg["target_fn"](df_val).values.astype(int)
        y_te = cfg["target_fn"](df_test).values.astype(int)

        # Remove any rows with invalid target (-1 or NaN-derived)
        tr_mask = y_tr >= 0
        va_mask = y_va >= 0
        te_mask = y_te >= 0

        X_tr_raw = df_train.loc[df_train.index[tr_mask], feat_cols].astype(float)
        X_va_raw = df_val.loc[df_val.index[va_mask], feat_cols].astype(float)
        X_te_raw = df_test.loc[df_test.index[te_mask], feat_cols].astype(float)
        y_tr = y_tr[tr_mask]
        y_va = y_va[va_mask]
        y_te = y_te[te_mask]

        imp = make_imputer(X_tr_raw)
        X_tr = apply_imputer(imp, X_tr_raw)
        X_va = apply_imputer(imp, X_va_raw)
        X_te = apply_imputer(imp, X_te_raw)

        sw = get_sample_weights(y_tr)
        model = xgb.XGBClassifier(
            num_class=n_cls,
            objective="multi:softmax",
            **xgb_default_params(),
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

        va_preds = model.predict(X_va)
        te_preds = model.predict(X_te)

        val_q  = qwk(y_va, va_preds)
        test_q = qwk(y_te, te_preds)
        val_f  = macro_f1(y_va, va_preds)
        test_f = macro_f1(y_te, te_preds)

        # Class distribution in each split
        for split_name, y_split in [("val", y_va), ("test", y_te), ("train", y_tr)]:
            unique, cnts = np.unique(y_split, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(unique, cnts)}
            print(f"    {split_name} distribution: {dist}")

        print(f"    val_QWK={val_q:.4f}  test_QWK={test_q:.4f}")
        print(f"    val_F1={val_f:.4f}   test_F1={test_f:.4f}")

        rows.append({
            "grouping": grouping_name,
            "n_classes": n_cls,
            "n_train": int(tr_mask.sum()),
            "n_val": int(va_mask.sum()),
            "n_test": int(te_mask.sum()),
            "val_qwk": val_q,
            "test_qwk": test_q,
            "val_macro_f1": val_f,
            "test_macro_f1": test_f,
        })

    out_df = pd.DataFrame(rows)
    out_path = DATA_PROC / "revision_target_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    out_df.to_parquet(DATA_PROC / "revision_target_sensitivity.parquet", index=False)
    print(f"\n  Saved to: {out_path}")

    return out_df


# ===========================================================================
# Final summary printer
# ===========================================================================

def print_summary(
    ablation_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    optuna_df: pd.DataFrame,
    ar_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> None:
    sep("=", 90)
    print("  REVISION ANALYSES - CONSOLIDATED SUMMARY")
    sep("=", 90)

    # Issue 1
    print("\n[ISSUE 1] Ablation Robustness")
    sep("-", 70)
    ordering_sub = ablation_df[ablation_df["analysis"] == "ordering"]
    for order in ordering_sub["order"].dropna().unique():
        retained = ordering_sub[
            (ordering_sub["order"] == order) & (ordering_sub["decision"] == "RETAIN")
        ]["group"].tolist()
        print(f"  {order:<30} retained groups: {retained}")

    boot_sub = ablation_df[ablation_df["analysis"] == "bootstrap_ci"].dropna(
        subset=["observed_delta"]
    )
    if not boot_sub.empty:
        print("\n  Bootstrap 95% CIs on QWK delta (Order A):")
        print(f"  {'Group':<20}  {'Obs delta':>10}  {'CI lower':>10}  {'CI upper':>10}")
        for _, r in boot_sub.iterrows():
            print(
                f"  {str(r['group']):<20}  {r['observed_delta']:>+10.4f}  "
                f"{r['ci_lower_95']:>+10.4f}  {r['ci_upper_95']:>+10.4f}"
            )

    perm_sub = ablation_df[ablation_df["analysis"] == "permutation_importance"].head(10)
    if not perm_sub.empty:
        print("\n  Top-10 features by permutation importance:")
        for _, r in perm_sub.iterrows():
            print(f"    {str(r['feature']):<35}  {r['perm_importance_mean']:+.4f}")

    # Issue 2
    print("\n[ISSUE 2] Val-Test Gap Diagnosis")
    sep("-", 70)
    dist_sub = gap_df[gap_df["analysis"] == "class_distribution"]
    if not dist_sub.empty:
        print("  Class distributions:")
        for split in ["train", "val", "test"]:
            rows = dist_sub[dist_sub["split"] == split]
            if not rows.empty:
                parts = " | ".join(
                    f"cls{int(r['class'])}={r['proportion']:.1%}" for _, r in rows.iterrows()
                )
                print(f"    {split:<6}: {parts}")

    rw_sub = gap_df[gap_df["analysis"] == "rolling_window"]
    if not rw_sub.empty:
        print(f"\n  Rolling-window QWK (15yr train, 5yr predict):")
        for _, r in rw_sub.iterrows():
            print(f"    {str(r['split']):<40}  QWK={r['metric_value']:.4f}")

    proto_sub = gap_df[gap_df["analysis"] == "protocol_comparison"]
    if not proto_sub.empty:
        print("\n  Protocol comparison:")
        for _, r in proto_sub.iterrows():
            val = r["metric_value"]
            if r["metric_name"] == "diagnosis_code":
                label = "CONCEPT_DRIFT" if val == 1.0 else "OVERFITTING_LIKELY"
                print(f"    {str(r['split']):<30}  {label}")
            else:
                print(f"    {str(r['split']):<30}  {r['metric_name']}={val:.4f}")

    # Issue 3
    print("\n[ISSUE 3] Nested CV Optuna vs. Original Protocol")
    sep("-", 70)
    print(f"  {'Protocol':<35}  {'val_QWK':>8}  {'test_QWK':>9}  {'val_F1':>7}  {'test_F1':>8}")
    for _, r in optuna_df.iterrows():
        print(
            f"  {str(r['protocol']):<35}  {r['val_qwk']:>8.4f}  "
            f"{r['test_qwk']:>9.4f}  {r['val_macro_f1']:>7.4f}  {r['test_macro_f1']:>8.4f}"
        )

    # Issue 4
    print("\n[ISSUE 4] Autoregressive Feature Analysis")
    sep("-", 70)
    print(f"  {'Model':<30}  {'n_feat':>6}  {'val_QWK':>8}  {'test_QWK':>9}  {'val_F1':>7}  {'test_F1':>8}")
    for _, r in ar_df.iterrows():
        vq = r["val_qwk"] if pd.notna(r["val_qwk"]) else float("nan")
        tq = r["test_qwk"] if pd.notna(r["test_qwk"]) else float("nan")
        vf = r["val_macro_f1"] if pd.notna(r.get("val_macro_f1")) else float("nan")
        tf = r["test_macro_f1"] if pd.notna(r.get("test_macro_f1")) else float("nan")
        print(
            f"  {str(r['model']):<30}  {int(r['n_features']) if pd.notna(r['n_features']) else '':>6}  "
            f"{vq:>8.4f}  {tq:>9.4f}  {vf:>7.4f}  {tf:>8.4f}"
        )

    # Issue 5
    print("\n[ISSUE 5] Target Grouping Sensitivity")
    sep("-", 70)
    print(f"  {'Grouping':<20}  {'n_cls':>6}  {'val_QWK':>8}  {'test_QWK':>9}  {'val_F1':>7}  {'test_F1':>8}")
    for _, r in target_df.iterrows():
        print(
            f"  {str(r['grouping']):<20}  {int(r['n_classes']):>6}  "
            f"{r['val_qwk']:>8.4f}  {r['test_qwk']:>9.4f}  "
            f"{r['val_macro_f1']:>7.4f}  {r['test_macro_f1']:>8.4f}"
        )

    sep("=", 90)
    print("  All revision results saved to data/processed/revision_*.csv and revision_*.parquet")
    sep("=", 90)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    sep("=", 90)
    print("  REVISION ANALYSES  -  Transboundary Water Conflict Prediction")
    sep("=", 90)

    print("\nLoading data ...", end=" ", flush=True)
    df = load_data()
    print(f"OK  ({len(df):,} events x {df.shape[1]} columns)")

    available = set(df.columns)
    df_train, df_val, df_test = make_splits(df)

    year_int = df["year"].fillna(-1).astype(int)
    print(
        f"Temporal split:  train={len(df_train):,} (<1996)  "
        f"val={len(df_val):,} (1996-2002)  "
        f"test={len(df_test):,} (>2002)"
    )

    print()

    ablation_df = issue1_ablation_robustness(df_train, df_val, df_test, available)
    gap_df      = issue2_gap_diagnosis(df_train, df_val, df_test, df, available)
    optuna_df   = issue3_nested_optuna(df_train, df_val, df_test, available)
    ar_df       = issue4_autoregressive(df_train, df_val, df_test, available)
    target_df   = issue5_target_sensitivity(df_train, df_val, df_test, available)

    print_summary(ablation_df, gap_df, optuna_df, ar_df, target_df)


if __name__ == "__main__":
    main()
