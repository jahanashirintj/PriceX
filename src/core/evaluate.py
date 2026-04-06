"""
evaluate.py
===========
Model evaluation: metrics, SHAP explainability, and PSI drift monitoring.

Functions:
  - compute_metrics   : RMSE, MAE, MAPE, R²
  - shap_global       : Beeswarm feature importance plot
  - shap_local        : Waterfall + force plot for a single prediction
  - psi_monitor       : Population Stability Index per feature
  - ks_drift_test     : KS-test on numeric feature distributions
  - full_report       : Runs all of the above and prints a summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
from scipy import stats as scipy_stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_space: bool = True,
    label: str = "",
) -> dict:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true    : Ground truth (log-price if log_space=True).
    y_pred    : Predictions (log-price if log_space=True).
    log_space : If True, inverse-transforms before computing metrics.

    Returns
    -------
    dict with rmse, mae, mape, r2.
    """
    if log_space:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    r2   = r2_score(y_true, y_pred)

    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Evaluation Metrics")
    print(f"  RMSE : {rmse:>12,.0f}")
    print(f"  MAE  : {mae:>12,.0f}")
    print(f"  MAPE : {mape:>11.2f}%")
    print(f"  R²   : {r2:>12.4f}")

    return dict(rmse=rmse, mae=mae, mape=mape, r2=r2)


# --------------------------------------------------------------------------
# SHAP Explainability
# --------------------------------------------------------------------------
def shap_global(
    model,
    X: pd.DataFrame,
    max_display: int = 15,
    save_path: str = None,
) -> np.ndarray:
    """
    Beeswarm plot — global feature importance across all predictions.

    Parameters
    ----------
    model       : Fitted ensemble or single tree model.
    X           : Feature DataFrame (sample for speed).
    max_display : How many features to show.
    save_path   : If provided, saves plot to this path.

    Returns
    -------
    shap_values array.
    """
    # Use a subsample for speed if dataset is large
    X_sample = X.sample(min(500, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Importance — Beeswarm", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] SHAP beeswarm saved to {save_path}")
    plt.show()

    return shap_values


def shap_local(
    model,
    X_row: pd.DataFrame,
    save_path: str = None,
) -> shap.Explanation:
    """
    Waterfall plot — explains a single prediction.

    Parameters
    ----------
    model   : Fitted model.
    X_row   : Single-row DataFrame.
    """
    explainer   = shap.TreeExplainer(model)
    explanation = explainer(X_row)

    # Waterfall
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(explanation[0], show=False)
    plt.title("SHAP Waterfall — Local Explanation", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".png", "_waterfall.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Force plot (HTML)
    force = shap.force_plot(
        explainer.expected_value,
        explanation[0].values,
        X_row,
        matplotlib=False,
    )
    if save_path:
        shap.save_html(save_path.replace(".png", "_force.html"), force)
        print(f"[evaluate] Force plot saved.")

    return explanation


def get_top_shap_factors(
    model,
    X_row: pd.DataFrame,
    n: int = 5,
) -> list:
    """
    Return top-n SHAP factors for a single prediction as a list of dicts.
    Used by the FastAPI /explain endpoint.
    """
    # StackingRegressor isn't directly supported by TreeExplainer.
    # We use the first base model (typically XGBoost) as an approximation.
    if hasattr(model, "estimators_"):
        model = model.estimators_[0]
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_row)
    pairs = list(zip(X_row.columns, shap_vals[0]))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in pairs[:n]]


# --------------------------------------------------------------------------
# Drift Monitoring
# --------------------------------------------------------------------------
def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """
    Population Stability Index.
    PSI < 0.10  → stable
    PSI 0.10–0.25 → monitor
    PSI > 0.25  → DRIFT — retrain
    """
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    breakpoints = np.linspace(lo, hi, buckets + 1)

    e_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    a_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Add small epsilon to avoid log(0)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)

    psi_value = np.sum((e_pct - a_pct) * np.log(e_pct / a_pct))
    return float(psi_value)


def psi_monitor(
    df_train: pd.DataFrame,
    df_new: pd.DataFrame,
    features: list,
) -> pd.DataFrame:
    """
    Compute PSI for every numeric feature.

    Returns
    -------
    DataFrame with columns: feature, psi, status
    """
    results = []
    for col in features:
        if col not in df_train.columns or col not in df_new.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_train[col]):
            continue
        psi_val = psi(
            df_train[col].dropna().values,
            df_new[col].dropna().values,
        )
        if psi_val < 0.10:
            status = "stable"
        elif psi_val < 0.25:
            status = "monitor"
        else:
            status = "DRIFT — retrain"
        results.append({"feature": col, "psi": round(psi_val, 4), "status": status})

    report = pd.DataFrame(results).sort_values("psi", ascending=False)
    print("\n[evaluate] PSI Drift Report:")
    print(report.to_string(index=False))
    return report


def ks_drift_test(
    df_train: pd.DataFrame,
    df_new: pd.DataFrame,
    features: list,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    KS-test for distributional drift on numeric features.

    Returns
    -------
    DataFrame with columns: feature, ks_stat, p_value, drifted
    """
    results = []
    for col in features:
        if col not in df_train.columns or col not in df_new.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_train[col]):
            continue
        stat, p_val = scipy_stats.ks_2samp(
            df_train[col].dropna().values,
            df_new[col].dropna().values,
        )
        results.append({
            "feature": col,
            "ks_stat": round(stat, 4),
            "p_value": round(p_val, 4),
            "drifted": p_val < alpha,
        })

    report = pd.DataFrame(results).sort_values("ks_stat", ascending=False)
    drifted = report[report["drifted"]]
    if not drifted.empty:
        print(f"\n[evaluate] KS-test: {len(drifted)} features show drift (p < {alpha}):")
        print(drifted[["feature","ks_stat","p_value"]].to_string(index=False))
    else:
        print(f"\n[evaluate] KS-test: No drift detected (p < {alpha}).")
    return report


# --------------------------------------------------------------------------
# Full evaluation report
# --------------------------------------------------------------------------
def full_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    listing_type: str,
    output_dir: str = ".",
) -> dict:
    """
    Run all evaluations and return a summary dict.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    log_pred = model.predict(X_test)
    metrics  = compute_metrics(y_test.values, log_pred, log_space=True, label=listing_type)

    shap_global(
        model, X_test,
        save_path=os.path.join(output_dir, f"{listing_type}_shap_beeswarm.png"),
    )
    shap_local(
        model, X_test.iloc[[0]],
        save_path=os.path.join(output_dir, f"{listing_type}_shap_local.png"),
    )

    drift_psi = psi_monitor(X_train, X_test, features=X_test.columns.tolist())
    drift_ks  = ks_drift_test(X_train, X_test, features=X_test.columns.tolist())

    # Log to MLflow
    with mlflow.start_run(run_name=f"{listing_type}_evaluation", nested=True):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(os.path.join(output_dir, f"{listing_type}_shap_beeswarm.png"))

    return {
        "metrics":   metrics,
        "psi_report": drift_psi,
        "ks_report":  drift_ks,
    }