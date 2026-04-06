"""
preprocess.py
=============
Stateful sklearn-compatible preprocessing transformer.
Fit on training data only — transform on train/val/test/prod.

Responsibilities:
  - Missing value imputation
  - Smoothed target encoding for locality (no leakage)
  - Binary flags and ordinal encodings
  - Derived numeric features
  - IQR-based outlier clipping
  - Log-transform on price target
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")


class HousePricePreprocessor(BaseEstimator, TransformerMixin):
    """
    Full preprocessing pipeline for house price data.

    Parameters
    ----------
    target_col    : Name of the price column.
    log_transform : Add a log1p-transformed target column.
    scale_numeric : Apply RobustScaler to numeric features.
    smoothing_k   : Smoothing factor for locality target encoding.
    """

    def __init__(
        self,
        target_col: str = "price",
        log_transform: bool = True,
        scale_numeric: bool = False,
        smoothing_k: int = 10,
    ):
        self.target_col = target_col
        self.log_transform = log_transform
        self.scale_numeric = scale_numeric
        self.smoothing_k = smoothing_k

    def fit(self, df: pd.DataFrame, y=None):
        df = df.copy()

        # --- Statistics for stateful imputation ---
        self.stats_ = {
            "area_sqft_median":         df["area_sqft"].median() if "area_sqft" in df.columns else 900,
            "bhk_mode":                 df["bhk"].mode()[0] if "bhk" in df.columns else 2,
            "age_mode":                 df["age_of_property"].mode()[0] if "age_of_property" in df.columns else 5,
            "locality_price_idx_median": 0, # To be filled below
            "locality_price_idx_p75":    0, # To be filled below
        }

        # Smoothed target encoding for locality
        # Formula: (count * locality_mean + k * global_mean) / (count + k)
        if self.target_col in df.columns:
            global_mean = df[self.target_col].mean()
            stats = (
                df.groupby("locality")[self.target_col]
                .agg(["mean", "count"])
                .rename(columns={"mean": "loc_mean", "count": "loc_count"})
            )
            k = self.smoothing_k
            stats["smoothed"] = (
                stats["loc_mean"] * stats["loc_count"] + global_mean * k
            ) / (stats["loc_count"] + k)
            self.locality_map_ = stats["smoothed"].to_dict()
            self.global_mean_  = global_mean
            
            # Store distribution stats for downstream engineering
            self.stats_["locality_price_idx_median"] = stats["smoothed"].median()
            self.stats_["locality_price_idx_p75"]    = stats["smoothed"].quantile(0.75)
        else:
            self.locality_map_ = {}
            self.global_mean_  = 5000000 # Fallback

        # Furnishing ordinal map
        self.furnish_map_ = {
            "Unfurnished":    0,
            "Semi-Furnished": 1,
            "Furnished":      2,
        }

        # IQR clip bounds (fit on training only)
        self.clip_bounds_ = {}
        for col in [self.target_col, "area_sqft"]:
            if col in df.columns:
                self.clip_bounds_[col] = (
                    df[col].quantile(0.01),
                    df[col].quantile(0.99),
                )

        # Optional scaler
        if self.scale_numeric:
            num_cols = [
                c for c in df.select_dtypes(include="number").columns
                if c != self.target_col
            ]
            self.scaler_ = RobustScaler()
            self.scaler_.fit(df[num_cols].fillna(0))
            self.scale_cols_ = num_cols
        else:
            self.scaler_ = None
            self.scale_cols_ = []

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Impute missing values ---
        df["area_sqft"]       = df["area_sqft"].fillna(self.stats_["area_sqft_median"])
        df["bhk"]             = df["bhk"].fillna(self.stats_["bhk_mode"])
        df["age_of_property"] = df["age_of_property"].fillna(self.stats_["age_mode"])
        df["furnishing"]      = df["furnishing"].fillna("Unfurnished")
        df["locality"]        = df["locality"].fillna("Unknown")
        df["location_type"]   = df["location_type"].fillna("URBAN")

        # --- Outlier clipping ---
        for col, (lo, hi) in self.clip_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        # --- Locality target encoding ---
        df["locality_price_idx"] = (
            df["locality"].map(self.locality_map_).fillna(self.global_mean_)
        )

        # --- Binary / ordinal flags ---
        df["is_urban"]          = (df["location_type"] == "URBAN").astype(int)
        df["is_furnished"]      = (df["furnishing"] == "Furnished").astype(int)
        df["is_semi_furnished"] = (df["furnishing"] == "Semi-Furnished").astype(int)
        df["furnished_score"]   = df["furnishing"].map(self.furnish_map_).fillna(0)

        # --- Derived numeric features ---
        df["log_area"]          = np.log1p(df["area_sqft"])
        df["bhk_area_ratio"]    = df["bhk"] / df["area_sqft"].clip(lower=1)
        df["age_x_bhk"]         = df["age_of_property"] * df["bhk"]
        df["urban_x_area"]      = df["is_urban"] * df["area_sqft"]
        df["bhk_squared"]       = df["bhk"] ** 2
        df["log_locality_idx"]  = np.log1p(df["locality_price_idx"])

        # Note: redundant features like size_tier and premium_location 
        # should be handled by specialized feature engineers to avoid duplication.

        # Note: price_per_sqft removed from production features to prevent leakage.
        # It can still be calculated for training labels if needed.

        # --- Log-transform target ---
        if self.log_transform and self.target_col in df.columns:
            df[f"log_{self.target_col}"] = np.log1p(df[self.target_col])

        # --- Optional scaling ---
        if self.scale_numeric and self.scaler_ is not None:
            available = [c for c in self.scale_cols_ if c in df.columns]
            df[available] = self.scaler_.transform(df[available].fillna(0))

        return df

    def get_feature_names(self, listing_type: str = "SALE") -> list:
        """Return ordered feature list for a given listing type."""
        try:
            from main import SALE_FEATURES, RENT_FEATURES
            return SALE_FEATURES if listing_type == "SALE" else RENT_FEATURES
        except ImportError:
            # Fallback if imported during training when main isn't in path
            if listing_type == "SALE":
                return [
                    "area_sqft", "bhk", "is_urban", "bhk_area_ratio", "urban_x_area", "bhk_squared", "age_x_bhk",
                    "log_area", "locality_price_idx", "log_locality_idx", "geo_cluster", "cluster_price_pct",
                    "age_of_property", "size_tier", "premium_location", "value_density"
                ]
            else:
                return [
                    "area_sqft", "bhk", "is_urban", "bhk_area_ratio", "urban_x_area", "bhk_squared", "age_x_bhk",
                    "log_area", "locality_price_idx", "log_locality_idx", "geo_cluster", "cluster_price_pct",
                    "furnished_score", "is_furnished", "is_semi_furnished", "bhk_x_furnish", "monthly_yield",
                    "furnish_x_urban", "small_unit_flag", "age_of_property"
                ]
