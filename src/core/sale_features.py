"""
sale_features.py
================
Sale-specific feature engineering applied AFTER shared preprocessing.

Derived features focus on capital value drivers:
  - Property size premium
  - Urban location uplift
  - Age depreciation
  - Price density interactions
"""

import numpy as np
import pandas as pd


# Feature list consumed by the sale model
SALE_FEATURES = [
    "area_sqft", "bhk", "is_urban", "bhk_area_ratio", "urban_x_area", "bhk_squared", "age_x_bhk",
    "log_area", "locality_price_idx", "log_locality_idx", "geo_cluster", "cluster_price_pct",
    "age_of_property", "size_tier", "premium_location", "value_density"
]


def engineer_sale(df: pd.DataFrame, stats: dict = None) -> pd.DataFrame:
    """
    Apply sale-specific feature engineering.

    Parameters
    ----------
    df    : Pre-processed DataFrame (output of HousePricePreprocessor.transform).
    stats : Optional stats dictionary from the preprocessor for stateful logic.

    Returns
    -------
    df with additional sale-specific columns.
    """
    df = df.copy()

    # --- Size tier (categorical bucket as int) ---
    # 0=Micro, 1=Small, 2=Medium, 3=Large, 4=Luxury
    bins   = [0, 400, 700, 1100, 1800, np.inf]
    labels = [0, 1, 2, 3, 4]
    df["size_tier"] = pd.cut(
        df["area_sqft"], bins=bins, labels=labels
    ).astype(int)

    # --- Premium location flag ---
    # Locality price index above 75th percentile of training distribution
    if stats and "locality_price_idx_p75" in stats:
        p75 = stats["locality_price_idx_p75"]
    else:
        p75 = df["locality_price_idx"].quantile(0.75) if len(df) > 1 else 0
    
    df["premium_location"] = (df["locality_price_idx"] >= p75).astype(int)

    # --- Value density: locality price index per unit area ---
    df["value_density"] = df["locality_price_idx"] / df["area_sqft"].clip(lower=1)

    # --- Age depreciation factor (newer = higher) ---
    df["age_factor"] = 1.0 / (1.0 + df["age_of_property"] * 0.02)

    # --- BHK premium: large BHK commands disproportionate premium ---
    df["bhk_premium"] = np.where(df["bhk"] >= 4, 1, 0)

    return df


def get_sale_X_y(df: pd.DataFrame):
    """
    Split into features and log-price target for model training.

    Returns
    -------
    X : pd.DataFrame with SALE_FEATURES columns
    y : pd.Series with log_price values
    """
    X = df.copy()
    for col in SALE_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[SALE_FEATURES]
    y = df["log_price"] if "log_price" in df.columns else np.log1p(df["price"])
    return X, y