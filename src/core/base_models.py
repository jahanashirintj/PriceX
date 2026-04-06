"""
base_models.py
==============
Factory functions for the three base learners used in the stacked ensemble.

Models:
  1. XGBoost  — gradient boosted trees, strong on tabular data
  2. LightGBM — faster training, good on high-cardinality features
  3. CatBoost — handles categoricals natively, robust to overfitting

Each factory accepts a params dict (from Optuna) and returns a
fitted-ready estimator with sensible defaults when params are absent.
"""

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.base import RegressorMixin


# --------------------------------------------------------------------------
# Default hyperparameters (before Optuna tuning)
# --------------------------------------------------------------------------
XGB_DEFAULTS = dict(
    n_estimators      = 600,
    max_depth         = 6,
    learning_rate     = 0.03,
    subsample         = 0.80,
    colsample_bytree  = 0.75,
    min_child_weight  = 3,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    gamma             = 0.0,
    random_state      = 42,
    verbosity         = 0,
    n_jobs            = -1,
    tree_method       = "hist",
)

LGB_DEFAULTS = dict(
    n_estimators      = 700,
    num_leaves        = 63,
    max_depth         = -1,
    learning_rate     = 0.025,
    feature_fraction  = 0.80,
    bagging_fraction  = 0.80,
    bagging_freq      = 5,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    verbose           = -1,
    n_jobs            = -1,
)

CAT_DEFAULTS = dict(
    iterations        = 500,
    depth             = 6,
    learning_rate     = 0.04,
    l2_leaf_reg       = 3.0,
    border_count      = 128,
    bagging_temperature = 1.0,
    random_seed       = 42,
    verbose           = 0,
    allow_writing_files = False,
)


# --------------------------------------------------------------------------
# Factory functions
# --------------------------------------------------------------------------
def make_xgb(params: dict = None) -> xgb.XGBRegressor:
    """Return an XGBoost regressor with given or default params."""
    cfg = {**XGB_DEFAULTS, **(params or {})}
    return xgb.XGBRegressor(**cfg)


def make_lgb(params: dict = None) -> lgb.LGBMRegressor:
    """Return a LightGBM regressor with given or default params."""
    cfg = {**LGB_DEFAULTS, **(params or {})}
    return lgb.LGBMRegressor(**cfg)


def make_cat(params: dict = None) -> CatBoostRegressor:
    """Return a CatBoost regressor with given or default params."""
    cfg = {**CAT_DEFAULTS, **(params or {})}
    return CatBoostRegressor(**cfg)


def get_base_models(params: dict = None) -> list:
    """
    Return the list of (name, estimator) tuples for StackingRegressor.

    Parameters
    ----------
    params : dict with keys like 'xgb_depth', 'lgb_leaves', 'cat_depth', etc.
             Typically the best_params dict from Optuna.

    Returns
    -------
    List of (str, estimator) tuples.
    """
    p = params or {}

    xgb_params = {
        "n_estimators":     p.get("xgb_n",      XGB_DEFAULTS["n_estimators"]),
        "max_depth":        p.get("xgb_depth",   XGB_DEFAULTS["max_depth"]),
        "learning_rate":    p.get("xgb_lr",      XGB_DEFAULTS["learning_rate"]),
        "subsample":        p.get("xgb_sub",     XGB_DEFAULTS["subsample"]),
        "colsample_bytree": p.get("xgb_col",     XGB_DEFAULTS["colsample_bytree"]),
        "reg_alpha":        p.get("xgb_alpha",   XGB_DEFAULTS["reg_alpha"]),
        "reg_lambda":       p.get("xgb_lambda",  XGB_DEFAULTS["reg_lambda"]),
    }
    lgb_params = {
        "n_estimators":    p.get("lgb_n",       LGB_DEFAULTS["n_estimators"]),
        "num_leaves":      p.get("lgb_leaves",  LGB_DEFAULTS["num_leaves"]),
        "learning_rate":   p.get("lgb_lr",      LGB_DEFAULTS["learning_rate"]),
        "feature_fraction":p.get("lgb_ff",      LGB_DEFAULTS["feature_fraction"]),
        "bagging_fraction":p.get("lgb_bf",      LGB_DEFAULTS["bagging_fraction"]),
    }
    cat_params = {
        "iterations":   p.get("cat_n",     CAT_DEFAULTS["iterations"]),
        "depth":        p.get("cat_depth", CAT_DEFAULTS["depth"]),
        "learning_rate":p.get("cat_lr",    CAT_DEFAULTS["learning_rate"]),
        "l2_leaf_reg":  p.get("cat_l2",   CAT_DEFAULTS["l2_leaf_reg"]),
    }

    return [
        ("xgb", make_xgb(xgb_params)),
        ("lgb", make_lgb(lgb_params)),
        ("cat", make_cat(cat_params)),
    ]


def model_info() -> None:
    """Print version info for all three libraries."""
    print(f"XGBoost  : {xgb.__version__}")
    print(f"LightGBM : {lgb.__version__}")
    import catboost
    print(f"CatBoost : {catboost.__version__}")


if __name__ == "__main__":
    model_info()
    models = get_base_models()
    for name, m in models:
        print(f"{name}: {type(m).__name__}")