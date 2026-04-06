"""
rent_features.py
================
Rent-specific feature engineering applied AFTER shared preprocessing.

Derived features focus on rental yield drivers:
  - Furnishing premium
  - Monthly yield density
  - Tenant demand proxies
  - Locality rental demand index
"""

import numpy as np
import pandas as pd


# Feature list consumed by the rent model
RENT_FEATURES = [
    "area_sqft", "bhk", "is_urban", "bhk_area_ratio", "urban_x_area", "bhk_squared", "age_x_bhk",
    "log_area", "locality_price_idx", "log_locality_idx", "geo_cluster", "cluster_price_pct",
    "furnished_score", "is_furnished", "is_semi_furnished", "bhk_x_furnish", "monthly_yield",
    "furnish_x_urban", "small_unit_flag", "age_of_property"
]


def engineer_rent(df: pd.DataFrame, stats: dict = None) -> pd.DataFrame:
    """
    Apply rent-specific feature engineering.

    Parameters
    ----------
    df    : Pre-processed DataFrame (output of HousePricePreprocessor.transform).
    stats : Optional stats dictionary from the preprocessor for stateful logic.

    Returns
    -------
    df with additional rent-specific columns.
    """
    df = df.copy()

    # --- Furnishing interaction features ---
    df["bhk_x_furnish"]  = df["bhk"] * df["furnished_score"]
    df["furnish_x_urban"] = df["furnished_score"] * df["is_urban"]

    # --- Monthly yield (rent per sqft) ---
    # In training, we use the actual price. In inference, we use the locality average.
    if "price" in df.columns:
        df["monthly_yield"] = df["price"] / df["area_sqft"].clip(lower=1)
    else:
        # Use locality average as a proxy for the 'typical' yield of this property
        df["monthly_yield"] = df["locality_price_idx"] / df["area_sqft"].clip(lower=1)

    # --- Small unit flag: 1BHK in urban = high rental demand ---
    df["small_unit_flag"] = (
        (df["bhk"] == 1) & (df["is_urban"] == 1)
    ).astype(int)

    # --- Locality rental demand index ---
    # Higher locality_price_idx + furnished = premium rental market
    df["rental_demand_idx"] = (
        df["locality_price_idx"] * (1 + 0.15 * df["furnished_score"])
    )

    # --- Effective area (furnished units feel larger) ---
    df["effective_area"] = df["area_sqft"] * (1 + 0.05 * df["furnished_score"])

    return df


def get_rent_X_y(df: pd.DataFrame):
    """
    Split into features and log-price target for model training.

    Returns
    -------
    X : pd.DataFrame with RENT_FEATURES columns
    y : pd.Series with log_price values
    """
    X = df.copy()
    for col in RENT_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[RENT_FEATURES]
    y = df["log_price"] if "log_price" in df.columns else np.log1p(df["price"])
    return X, y