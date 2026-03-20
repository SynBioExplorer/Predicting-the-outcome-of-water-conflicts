#!/usr/bin/env python3
"""
Autoresearch-inspired feature ablation for transboundary water conflict prediction.

Fixed evaluation protocol:
    - Temporal split: train <1996, val 1996-2002, test >2002
    - Model: LightGBM with fixed hyperparameters
    - Metric for retain/discard: Quadratic Weighted Kappa (QWK) on validation set
    - Retain threshold: delta QWK >= 0.005

Feature groups are tested incrementally; each group is retained only if it
improves validation QWK over the running best.

Run with:
    conda run -n water-conflict python3 scripts/02_ablation.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ = Path("/Users/felix/Documents/Predicting-the-outcome-of-water-conflicts")
DATA_PROC = PROJ / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "baseline_tfdd": {
        "desc": "TFDD BCU features only",
        "cols": [
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
    },
    "+climate": {
        "desc": "Add basin-level climate",
        "cols": ["pre", "pet", "spei", "pre_anomaly", "pre_ltm"],
    },
    "+governance": {
        "desc": "Add Polity + WGI",
        "cols": [
            "polity2_pol1", "polity2_pol2",
            "RL.EST_wgi1", "RL.EST_wgi2",
            "PV.EST_wgi1", "PV.EST_wgi2",
            "GE.EST_wgi1", "GE.EST_wgi2",
            "CC.EST_wgi1", "CC.EST_wgi2",
        ],
    },
    "+economic": {
        "desc": "Add GDP, military, population",
        "cols": [
            "NY.GDP.PCAP.CD_wdi1", "NY.GDP.PCAP.CD_wdi2",
            "MS.MIL.XPND.GD.ZS_wdi1", "MS.MIL.XPND.GD.ZS_wdi2",
            "SP.POP.TOTL_wdi1", "SP.POP.TOTL_wdi2",
            "ER.H2O.FWTL.ZS_wdi1", "ER.H2O.FWTL.ZS_wdi2",
            "ER.H2O.INTR.PC_wdi1", "ER.H2O.INTR.PC_wdi2",
        ],
    },
    "+aquastat": {
        "desc": "Add AQUASTAT water dependency",
        "cols": [
            "aq_fwtl_zs_aq1", "aq_fwtl_zs_aq2",
            "aq_intr_pc_aq1", "aq_intr_pc_aq2",
            "aq_fwag_zs_aq1", "aq_fwag_zs_aq2",
        ],
    },
    "+asymmetry": {
        "desc": "Add dyadic asymmetry features",
        "cols": [
            "pop_ratio", "withdrawal_ratio", "dam_ratio",
            "instit_vuln_diff", "hydropol_max",
            "gdp_ratio", "polity_diff", "water_stress_diff",
        ],
    },
    "+temporal": {
        "desc": "Add temporal features",
        "cols": [
            "events_prior_5yr", "cooperation_momentum", "cold_war",
            "treaty_rate_5yr", "event_escalation", "year",
        ],
    },
}

# ---------------------------------------------------------------------------
# Fixed LightGBM hyperparameters
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    class_weight="balanced",
    verbose=-1,
    random_state=42,
)

# Minimum QWK improvement to retain a feature group
RETAIN_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_cols(requested_cols: list[str], available: set[str]) -> list[str]:
    """Return only the columns present in the dataset; warn about missing ones."""
    kept, dropped = [], []
    for c in requested_cols:
        if c in available:
            kept.append(c)
        else:
            dropped.append(c)
    if dropped:
        print(f"  [warn] columns not found in data (skipped): {dropped}")
    return kept


def encode_issue_type(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Issue_Type1 float codes to integer category codes."""
    df = df.copy()
    if "Issue_Type1" in df.columns:
        df["Issue_Type1"] = df["Issue_Type1"].astype("Int64")
    return df


def impute_with_train_medians(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit median imputation on the training set; apply to val and test.
    Only imputes features with >60% missingness in training data.
    LightGBM handles moderate NaN natively, so we only intervene for
    very high-missingness columns to avoid introducing spurious signal.
    """
    HIGH_MISS_THRESH = 0.60
    train_miss = X_train.isnull().mean()
    high_miss_cols = train_miss[train_miss > HIGH_MISS_THRESH].index.tolist()

    if high_miss_cols:
        medians = X_train[high_miss_cols].median()
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        for col in high_miss_cols:
            X_train[col] = X_train[col].fillna(medians[col])
            X_val[col] = X_val[col].fillna(medians[col])
            X_test[col] = X_test[col].fillna(medians[col])

    return X_train, X_val, X_test


def compute_metrics(y_true, y_pred) -> dict:
    """Compute QWK, macro-F1, accuracy, and MAE."""
    return {
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def train_and_evaluate(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, dict, np.ndarray]:
    """
    Train LightGBM on df_train, evaluate on df_val and df_test.
    Returns (val_metrics, test_metrics, test_predictions).
    """
    X_train = df_train[feature_cols].copy()
    X_val   = df_val[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()

    y_train = df_train["target"].astype(int)
    y_val   = df_val["target"].astype(int)
    y_test  = df_test["target"].astype(int)

    # Apply high-missingness imputation (fitted on train)
    X_train, X_val, X_test = impute_with_train_medians(X_train, X_val, X_test)

    # Ensure all numeric (Issue_Type1 is Int64 -> cast to float for lgbm)
    X_train = X_train.astype(float)
    X_val   = X_val.astype(float)
    X_test  = X_test.astype(float)

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    val_preds  = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_metrics  = compute_metrics(y_val, val_preds)
    test_metrics = compute_metrics(y_test, test_preds)

    return val_metrics, test_metrics, test_preds, model


def print_separator(char: str = "-", width: int = 100) -> None:
    print(char * width)


def format_table_row(
    group: str,
    desc: str,
    n_feat: int,
    val_qwk: float,
    val_f1: float,
    val_acc: float,
    delta_qwk: float | str,
    decision: str,
) -> str:
    delta_str = f"{delta_qwk:+.4f}" if isinstance(delta_qwk, float) else str(delta_qwk)
    return (
        f"  {group:<18} | {desc:<40} | {n_feat:>6} | "
        f"{val_qwk:.4f} | {val_f1:.4f} | {val_acc:.4f} | "
        f"{delta_str:>8} | {decision}"
    )


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

def main() -> None:
    print_separator("=")
    print("  AUTORESEARCH-INSPIRED FEATURE ABLATION")
    print("  Transboundary Water Conflict Outcome Prediction")
    print_separator("=")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading data ...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PROC / "events_enriched.parquet")
    df = encode_issue_type(df)
    print(f"OK  ({len(df):,} events x {df.shape[1]} columns)")

    available_cols = set(df.columns)

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------
    df_train = df[df["year"] < 1996].copy()
    df_val   = df[(df["year"] >= 1996) & (df["year"] <= 2002)].copy()
    df_test  = df[df["year"] > 2002].copy()

    print(
        f"\nTemporal split:  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}"
    )
    print(
        f"  Train years: {int(df_train['year'].min())}-{int(df_train['year'].max())}  "
        f"Val years: {int(df_val['year'].min())}-{int(df_val['year'].max())}  "
        f"Test years: {int(df_test['year'].min())}-{int(df_test['year'].max())}"
    )

    # ------------------------------------------------------------------
    # Ablation state
    # ------------------------------------------------------------------
    retained_cols: list[str] = []     # cumulative retained feature columns
    best_val_qwk: float = -999.0
    results: list[dict] = []

    group_names = list(FEATURE_GROUPS.keys())

    print_separator()
    print(
        f"  {'Group':<18} | {'Description':<40} | {'#Feat':>6} | "
        f"{'ValQWK':>7} | {'ValF1':>7} | {'ValAcc':>7} | "
        f"{'DeltaQWK':>8} | Decision"
    )
    print_separator()

    for i, group_name in enumerate(group_names):
        group = FEATURE_GROUPS[group_name]
        candidate_cols = filter_cols(group["cols"], available_cols)

        if i == 0:
            # Baseline: train only on this group
            trial_cols = candidate_cols
        else:
            # Incremental: add this group to retained set
            trial_cols = retained_cols + candidate_cols

        # Deduplicate while preserving order
        seen = set()
        trial_cols_dedup = []
        for c in trial_cols:
            if c not in seen:
                seen.add(c)
                trial_cols_dedup.append(c)
        trial_cols = trial_cols_dedup

        val_metrics, test_metrics, test_preds, _ = train_and_evaluate(
            df_train, df_val, df_test, trial_cols
        )

        val_qwk = val_metrics["qwk"]
        delta = val_qwk - best_val_qwk

        if i == 0:
            # Baseline always retained
            decision = "RETAIN (baseline)"
            retained_cols = list(trial_cols)
            best_val_qwk = val_qwk
            delta_display: float | str = "N/A"
        else:
            if delta >= RETAIN_THRESHOLD:
                decision = "RETAIN"
                retained_cols = list(trial_cols)
                best_val_qwk = val_qwk
            else:
                decision = "DISCARD"
            delta_display = delta

        row = {
            "group": group_name,
            "desc": group["desc"],
            "n_features": len(trial_cols),
            "val_qwk": val_qwk,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_mae": val_metrics["mae"],
            "delta_qwk": delta_display,
            "decision": decision,
            "retained_cols": ";".join(trial_cols) if decision != "DISCARD" else "",
        }
        results.append(row)

        print(
            format_table_row(
                group_name,
                group["desc"],
                len(trial_cols),
                val_qwk,
                val_metrics["macro_f1"],
                val_metrics["accuracy"],
                delta_display,
                decision,
            )
        )

    print_separator()

    # ------------------------------------------------------------------
    # Final model: all retained features, evaluate on held-out test set
    # ------------------------------------------------------------------
    print(f"\nFinal retained feature set: {len(retained_cols)} features")
    print(f"  {retained_cols}")

    print("\nTraining final model on all retained features ...")
    final_val_metrics, final_test_metrics, final_test_preds, final_model = train_and_evaluate(
        df_train, df_val, df_test, retained_cols
    )

    print_separator("=")
    print("  FINAL MODEL PERFORMANCE")
    print_separator("=")
    print(f"  {'Split':<10}  {'QWK':>7}  {'MacroF1':>8}  {'Accuracy':>9}  {'MAE':>6}")
    print_separator("-", 55)
    m = final_val_metrics
    print(
        f"  {'Val':<10}  {m['qwk']:>7.4f}  {m['macro_f1']:>8.4f}  "
        f"{m['accuracy']:>9.4f}  {m['mae']:>6.4f}"
    )
    m = final_test_metrics
    print(
        f"  {'Test':<10}  {m['qwk']:>7.4f}  {m['macro_f1']:>8.4f}  "
        f"{m['accuracy']:>9.4f}  {m['mae']:>6.4f}"
    )
    print_separator("=")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # Ablation results CSV (exclude the big retained_cols string)
    ablation_df = pd.DataFrame(results)[
        ["group", "desc", "n_features", "val_qwk", "val_macro_f1",
         "val_accuracy", "val_mae", "delta_qwk", "decision"]
    ]
    ablation_path = DATA_PROC / "ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\nAblation table saved to: {ablation_path}")

    # Test predictions parquet
    test_pred_df = df_test[["ID", "year", "target"]].copy()
    test_pred_df["predicted"] = final_test_preds
    test_pred_df["correct"] = (test_pred_df["predicted"] == test_pred_df["target"].astype(int))
    test_pred_path = DATA_PROC / "test_predictions.parquet"
    test_pred_df.to_parquet(test_pred_path, index=False)
    print(f"Test predictions saved to: {test_pred_path}")

    # ------------------------------------------------------------------
    # Print full ablation table at end for easy review
    # ------------------------------------------------------------------
    print_separator("=")
    print("  FULL ABLATION TABLE")
    print_separator("=")
    print(
        f"  {'Group':<18} | {'#Feat':>6} | "
        f"{'ValQWK':>7} | {'ValF1':>7} | {'ValAcc':>7} | "
        f"{'DeltaQWK':>8} | Decision"
    )
    print_separator()
    for r in results:
        delta_str = (
            f"{r['delta_qwk']:+.4f}"
            if isinstance(r["delta_qwk"], float)
            else str(r["delta_qwk"])
        )
        print(
            f"  {r['group']:<18} | {r['n_features']:>6} | "
            f"{r['val_qwk']:.4f} | {r['val_macro_f1']:.4f} | {r['val_accuracy']:.4f} | "
            f"{delta_str:>8} | {r['decision']}"
        )
    print_separator()
    print(f"\n  Best val QWK (retained): {best_val_qwk:.4f}")
    print(f"  Final test QWK:          {final_test_metrics['qwk']:.4f}")
    print(f"  Final test macro-F1:     {final_test_metrics['macro_f1']:.4f}")
    print(f"  Final test accuracy:     {final_test_metrics['accuracy']:.4f}")
    print(f"  Final test MAE:          {final_test_metrics['mae']:.4f}")
    print_separator("=")


if __name__ == "__main__":
    main()
