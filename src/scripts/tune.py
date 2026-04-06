"""
tune.py
=======
Optuna-powered hyperparameter optimisation for the stacked ensemble.

Runs 80 trials per listing type (sale / rent) using:
  - MedianPruner: kills unpromising trials early
  - 5-fold CV with MAPE objective
  - MLflow auto-logging for every trial

Usage:
    python tune.py --listing sale --trials 80
    python tune.py --listing rent --trials 80
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------
# Objective function
# --------------------------------------------------------------------------
def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optuna objective — returns mean CV MAPE (lower is better).
    All hyperparameters drawn from their search spaces.
    """
    # XGBoost params
    xgb_params = dict(
        n_estimators     = trial.suggest_int("xgb_n",     200, 800),
        max_depth        = trial.suggest_int("xgb_depth", 4,   8),
        learning_rate    = trial.suggest_float("xgb_lr",  0.01, 0.15, log=True),
        subsample        = trial.suggest_float("xgb_sub", 0.60, 1.00),
        colsample_bytree = trial.suggest_float("xgb_col", 0.60, 1.00),
        reg_alpha        = trial.suggest_float("xgb_alpha",  1e-3, 1.0, log=True),
        reg_lambda       = trial.suggest_float("xgb_lambda", 0.5,  5.0),
        random_state     = 42, verbosity=0, n_jobs=-1, tree_method="hist",
    )

    # LightGBM params
    lgb_params = dict(
        n_estimators     = trial.suggest_int("lgb_n",       200, 900),
        num_leaves       = trial.suggest_int("lgb_leaves",  31,  127),
        learning_rate    = trial.suggest_float("lgb_lr",    0.01, 0.15, log=True),
        feature_fraction = trial.suggest_float("lgb_ff",    0.60, 1.00),
        bagging_fraction = trial.suggest_float("lgb_bf",    0.60, 1.00),
        min_child_samples= trial.suggest_int("lgb_min",     10,   50),
        reg_alpha        = trial.suggest_float("lgb_alpha", 1e-3, 1.0, log=True),
        random_state     = 42, verbose=-1, n_jobs=-1,
    )

    # CatBoost params
    cat_params = dict(
        iterations    = trial.suggest_int("cat_n",     200, 700),
        depth         = trial.suggest_int("cat_depth", 4,   8),
        learning_rate = trial.suggest_float("cat_lr",  0.01, 0.15, log=True),
        l2_leaf_reg   = trial.suggest_float("cat_l2",  1.0, 10.0),
        bagging_temperature = trial.suggest_float("cat_bag", 0.0, 1.0),
        random_seed   = 42, verbose=0, allow_writing_files=False,
    )

    # Meta-learner
    meta_alpha = trial.suggest_float("meta_alpha", 0.01, 20.0, log=True)

    # Build stacker
    base = [
        ("xgb", xgb.XGBRegressor(**xgb_params)),
        ("lgb", lgb.LGBMRegressor(**lgb_params)),
        ("cat", CatBoostRegressor(**cat_params)),
    ]
    stack = StackingRegressor(
        estimators      = base,
        final_estimator = Ridge(alpha=meta_alpha),
        cv              = 5,
        passthrough     = True,
        n_jobs          = -1,
    )

    # 5-fold CV (log-price space, MAPE after inverse transform)
    scores = cross_val_score(
        stack, X, y,
        scoring  = "neg_mean_absolute_percentage_error",
        cv       = 5,
        n_jobs   = -1,
        error_score = "raise",
    )
    mape = -scores.mean()

    # Log to MLflow
    with mlflow.start_run(nested=True):
        all_params = {**xgb_params, **lgb_params, **cat_params, "meta_alpha": meta_alpha}
        mlflow.log_params({k: round(v, 6) if isinstance(v, float) else v
                           for k, v in all_params.items()})
        mlflow.log_metric("cv_mape", mape)

    return mape


# --------------------------------------------------------------------------
# Main tuning runner
# --------------------------------------------------------------------------
def run_hpo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    listing_type: str,
    n_trials: int = 80,
    out_dir: str = "../../models",
) -> dict:
    """
    Run Optuna HPO and return the best hyperparameter dict.

    Parameters
    ----------
    listing_type : "sale" or "rent"
    n_trials     : Number of Optuna trials.
    out_dir      : Where to save the best_params YAML / pkl.

    Returns
    -------
    best_params : dict of best hyperparameters.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Optuna HPO — {listing_type.upper()}  ({n_trials} trials)")
    print(f"{'='*60}\n")

    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    with mlflow.start_run(run_name=f"{listing_type}_hpo"):
        study = optuna.create_study(
            direction  = "minimize",
            sampler    = sampler,
            pruner     = pruner,
            study_name = f"{listing_type}_hpo",
        )
        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials         = n_trials,
            show_progress_bar= True,
            catch            = (Exception,),
        )

        best = study.best_params
        best_mape = study.best_value

        print(f"\n[tune] Best MAPE : {best_mape:.4f}")
        print(f"[tune] Best params: {best}")

        mlflow.log_metric("best_mape", best_mape)
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})

    # Save best params
    params_path = os.path.join(out_dir, f"{listing_type}_best_params.pkl")
    joblib.dump(best, params_path)
    print(f"[tune] Best params saved to {params_path}")

    # Optuna visualisation summary
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(out_dir, f"{listing_type}_param_importance.html"))
    except Exception:
        pass

    return best


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", choices=["sale", "rent"], required=True)
    parser.add_argument("--trials",  type=int, default=80)
    parser.add_argument("--proc_dir", default="../../data/processed")
    parser.add_argument("--out_dir",  default="../../models")
    args = parser.parse_args()

    # Load processed features
    train_path = os.path.join(args.proc_dir, f"{args.listing}_train.csv")
    df_train   = pd.read_csv(train_path)

    sys.path.insert(0, str(Path(__file__).parent.parent / "features"))
    if args.listing == "sale":
        from sale_features import SALE_FEATURES, get_sale_X_y
        X_train, y_train = get_sale_X_y(df_train)
    else:
        from rent_features import RENT_FEATURES, get_rent_X_y
        X_train, y_train = get_rent_X_y(df_train)

    best_params = run_hpo(
        X_train, y_train,
        listing_type = args.listing,
        n_trials     = args.trials,
        out_dir      = args.out_dir,
    )
    print("\n[tune] Done.")