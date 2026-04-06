"""
stacker.py
==========
Builds, trains, evaluates, and persists the stacked ensemble.

Architecture:
  Layer 1 (base learners): XGBoost, LightGBM, CatBoost
      — trained via 5-fold cross-validation, producing out-of-fold predictions
  Layer 2 (meta-learner) : Ridge regression
      — learns how to best combine base learner predictions
  passthrough=True        — meta-learner also sees original features

Usage:
    from stacker import build_stacker, train_and_save
    model = train_and_save(X_train, y_train, X_test, y_test, "sale")
"""

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from base_models import get_base_models


# --------------------------------------------------------------------------
def build_stacker(
    params: dict = None,
    meta_alpha: float = 1.0,
    cv: int = 5,
) -> StackingRegressor:
    """
    Construct the StackingRegressor with tuned or default base models.

    Parameters
    ----------
    params     : Optuna best_params dict (passed to get_base_models).
    meta_alpha : Regularisation for Ridge meta-learner.
    cv         : Number of CV folds for out-of-fold predictions.

    Returns
    -------
    Unfitted StackingRegressor.
    """
    base_models = get_base_models(params)
    meta        = Ridge(alpha=meta_alpha)

    stacker = StackingRegressor(
        estimators      = base_models,
        final_estimator = meta,
        cv              = cv,
        passthrough     = True,   # meta-learner sees raw features too
        n_jobs          = -1,
    )
    return stacker


# --------------------------------------------------------------------------
def cv_evaluate(
    model: StackingRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
) -> dict:
    """
    Run n-fold CV and return mean MAPE, RMSE, R².

    Note: predictions are in log-price space — metrics computed
    after inverse-transforming to raw price.
    """
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    maes, rmses, mapes, r2s = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        log_pred = model.predict(X_val)

        # Inverse log transform
        y_true_raw = np.expm1(y_val)
        y_pred_raw = np.expm1(log_pred)

        rmses.append(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
        maes.append(mean_absolute_error(y_true_raw, y_pred_raw))
        mapes.append(np.mean(np.abs((y_true_raw - y_pred_raw) / y_true_raw.clip(1))) * 100)
        r2s.append(r2_score(y_true_raw, y_pred_raw))
        print(f"  Fold {fold}: MAPE={mapes[-1]:.2f}%  RMSE={rmses[-1]:,.0f}  R²={r2s[-1]:.4f}")

    results = dict(
        mape=np.mean(mapes),
        rmse=np.mean(rmses),
        mae =np.mean(maes),
        r2  =np.mean(r2s),
    )
    print(f"\n  CV Mean — MAPE: {results['mape']:.2f}%  RMSE: {results['rmse']:,.0f}"
          f"  MAE: {results['mae']:,.0f}  R²: {results['r2']:.4f}")
    return results


# --------------------------------------------------------------------------
def train_and_save(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    listing_type: str,
    params: dict = None,
    out_dir: str = "../../models",
    run_cv: bool = True,
) -> StackingRegressor:
    """
    Full train → evaluate → save pipeline for one listing type.

    Parameters
    ----------
    listing_type : "sale" or "rent"
    params       : Optuna best_params (None = use defaults)
    out_dir      : Where to save the .pkl file
    run_cv       : Whether to run cross-validation before final fit

    Returns
    -------
    Fitted StackingRegressor.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_alpha = params.pop("meta_alpha", 1.0) if params else 1.0
    model = build_stacker(params=params, meta_alpha=meta_alpha)

    print(f"\n{'='*60}")
    print(f" Training {listing_type.upper()} stacked ensemble")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=f"{listing_type}_stacker"):
        if run_cv:
            print("\n[stacker] Running 5-fold CV...")
            cv_results = cv_evaluate(model, X_train, y_train)
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_results.items()})

        # Final fit on full training set
        print("\n[stacker] Final fit on full training data...")
        model.fit(X_train, y_train)

        # Test set evaluation
        log_pred   = model.predict(X_test)
        y_true_raw = np.expm1(y_test)
        y_pred_raw = np.expm1(log_pred)

        test_rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
        test_mae  = mean_absolute_error(y_true_raw, y_pred_raw)
        test_mape = np.mean(np.abs((y_true_raw - y_pred_raw) / y_true_raw.clip(1))) * 100
        test_r2   = r2_score(y_true_raw, y_pred_raw)

        print(f"\n[stacker] Test set — MAPE: {test_mape:.2f}%  "
              f"RMSE: {test_rmse:,.0f}  MAE: {test_mae:,.0f}  R²: {test_r2:.4f}")

        mlflow.log_metrics({
            "test_rmse": test_rmse,
            "test_mae":  test_mae,
            "test_mape": test_mape,
            "test_r2":   test_r2,
        })

        if params:
            mlflow.log_params(params)

        # Save
        model_path = os.path.join(out_dir, f"{listing_type}_model.pkl")
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path=f"{listing_type}_model")
        print(f"\n[stacker] Model saved to {model_path}")

    return model


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test with synthetic data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(np.log1p(np.abs(y) * 1000))

    split = int(0.8 * len(X))
    model = train_and_save(
        X.iloc[:split], y.iloc[:split],
        X.iloc[split:], y.iloc[split:],
        listing_type="test",
        out_dir="/tmp",
        run_cv=False,
    )
    print("\nSmoke test passed.")