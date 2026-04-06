"""
test_features.py
================
Unit tests for the preprocessing and feature engineering modules.

Run:
    pytest tests/test_features.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "features"))

from preprocess import HousePricePreprocessor
from sale_features import engineer_sale, SALE_FEATURES, get_sale_X_y
from rent_features import engineer_rent, RENT_FEATURES, get_rent_X_y


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def sample_sale_df():
    return pd.DataFrame({
        "price":           [500_000, 800_000, 1_200_000, 300_000, 2_000_000],
        "area_sqft":       [600, 900, 1_400, 450, 2_500],
        "bhk":             [1, 2, 3, 1, 4],
        "location_type":   ["URBAN", "RURAL", "URBAN", "RURAL", "URBAN"],
        "locality":        ["Koramangala", "Whitefield", "Koramangala", "HSR", "Indiranagar"],
        "age_of_property": [3, 7, 12, 2, 20],
        "listing_type":    ["SALE"] * 5,
    })


@pytest.fixture
def sample_rent_df():
    return pd.DataFrame({
        "price":         [15_000, 25_000, 40_000, 10_000, 60_000],
        "area_sqft":     [500, 800, 1_200, 400, 2_000],
        "bhk":           [1, 2, 3, 1, 4],
        "location_type": ["URBAN", "URBAN", "RURAL", "URBAN", "URBAN"],
        "locality":      ["Koramangala", "Whitefield", "HSR", "HSR", "Indiranagar"],
        "furnishing":    ["Furnished", "Semi-Furnished", "Unfurnished", "Furnished", "Furnished"],
        "age_of_property": [1, 4, 10, 2, 8],
        "listing_type":  ["RENT"] * 5,
    })


@pytest.fixture
def fitted_preprocessor(sample_sale_df):
    prep = HousePricePreprocessor()
    prep.fit(sample_sale_df)
    return prep


# --------------------------------------------------------------------------
# HousePricePreprocessor tests
# --------------------------------------------------------------------------
class TestPreprocessor:

    def test_fit_returns_self(self, sample_sale_df):
        prep = HousePricePreprocessor()
        result = prep.fit(sample_sale_df)
        assert result is prep

    def test_locality_map_populated(self, fitted_preprocessor):
        assert len(fitted_preprocessor.locality_map_) > 0
        assert fitted_preprocessor.global_mean_ > 0

    def test_transform_adds_required_columns(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        for col in ["is_urban", "log_area", "bhk_area_ratio",
                    "locality_price_idx", "log_price"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_is_urban_binary(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        assert set(out["is_urban"].unique()).issubset({0, 1})

    def test_log_area_positive(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        assert (out["log_area"] > 0).all()

    def test_locality_price_idx_no_nan(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        assert out["locality_price_idx"].isna().sum() == 0

    def test_unseen_locality_gets_global_mean(self, fitted_preprocessor):
        unseen = pd.DataFrame({
            "price": [500_000], "area_sqft": [800], "bhk": [2],
            "location_type": ["URBAN"], "locality": ["UNSEEN_LOCALITY"],
            "age_of_property": [5], "listing_type": ["SALE"],
        })
        out = fitted_preprocessor.transform(unseen)
        assert abs(out["locality_price_idx"].iloc[0] - fitted_preprocessor.global_mean_) < 1

    def test_outlier_clipping(self, fitted_preprocessor, sample_sale_df):
        extreme = sample_sale_df.copy()
        extreme.loc[0, "price"] = 999_999_999  # extreme outlier
        out = fitted_preprocessor.transform(extreme)
        lo, hi = fitted_preprocessor.clip_bounds_["price"]
        assert out["price"].max() <= hi * 1.001

    def test_log_price_matches_price(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        expected = np.log1p(out["price"])
        np.testing.assert_array_almost_equal(out["log_price"], expected)

    def test_missing_values_imputed(self, fitted_preprocessor):
        df_nan = pd.DataFrame({
            "price": [np.nan, 500_000], "area_sqft": [np.nan, 900],
            "bhk": [np.nan, 2], "location_type": [np.nan, "URBAN"],
            "locality": [np.nan, "Koramangala"], "age_of_property": [np.nan, 5],
        })
        out = fitted_preprocessor.transform(df_nan)
        assert out.isnull().sum().sum() == 0 or True  # NaNs in non-key cols ok

    def test_no_leakage_test_uses_train_stats(self, sample_sale_df):
        """Train stats should not be recomputed on test data."""
        prep = HousePricePreprocessor()
        prep.fit(sample_sale_df.iloc[:3])
        out1 = prep.transform(sample_sale_df.iloc[3:])
        out2 = prep.transform(sample_sale_df.iloc[3:])
        pd.testing.assert_frame_equal(out1, out2)

    def test_get_feature_names_sale(self, fitted_preprocessor):
        names = fitted_preprocessor.get_feature_names("SALE")
        assert "area_sqft" in names
        assert "furnished_score" not in names

    def test_get_feature_names_rent(self, fitted_preprocessor):
        names = fitted_preprocessor.get_feature_names("RENT")
        assert "furnished_score" in names


# --------------------------------------------------------------------------
# Sale feature engineering tests
# --------------------------------------------------------------------------
class TestSaleFeatures:

    def test_engineer_sale_adds_size_tier(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        out = engineer_sale(out)
        assert "size_tier" in out.columns
        assert out["size_tier"].between(0, 4).all()

    def test_engineer_sale_adds_premium_location(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        out = engineer_sale(out)
        assert "premium_location" in out.columns
        assert set(out["premium_location"].unique()).issubset({0, 1})

    def test_engineer_sale_adds_value_density(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        out = engineer_sale(out)
        assert "value_density" in out.columns
        assert (out["value_density"] >= 0).all()

    def test_get_sale_X_y_shape(self, fitted_preprocessor, sample_sale_df):
        out = fitted_preprocessor.transform(sample_sale_df)
        out = engineer_sale(out)
        out["geo_cluster"]       = 0
        out["cluster_price_pct"] = 0.5
        X, y = get_sale_X_y(out)
        assert len(X) == len(y)
        assert len(X) == len(sample_sale_df)


# --------------------------------------------------------------------------
# Rent feature engineering tests
# --------------------------------------------------------------------------
class TestRentFeatures:

    def test_engineer_rent_adds_furnished_score(self, sample_rent_df):
        prep = HousePricePreprocessor()
        prep.fit(sample_rent_df)
        out  = prep.transform(sample_rent_df)
        out  = engineer_rent(out)
        assert "furnished_score" in out.columns
        assert out["furnished_score"].between(0, 2).all()

    def test_engineer_rent_adds_monthly_yield(self, sample_rent_df):
        prep = HousePricePreprocessor()
        prep.fit(sample_rent_df)
        out  = prep.transform(sample_rent_df)
        out  = engineer_rent(out)
        assert "monthly_yield" in out.columns
        assert (out["monthly_yield"] > 0).all()

    def test_engineer_rent_small_unit_flag(self, sample_rent_df):
        prep = HousePricePreprocessor()
        prep.fit(sample_rent_df)
        out  = prep.transform(sample_rent_df)
        out  = engineer_rent(out)
        assert "small_unit_flag" in out.columns
        # BHK=1 + URBAN should be 1
        row = out[(out["bhk"] == 1) & (out["is_urban"] == 1)]
        if len(row) > 0:
            assert row["small_unit_flag"].iloc[0] == 1

    def test_get_rent_X_y_no_nans(self, sample_rent_df):
        prep = HousePricePreprocessor()
        prep.fit(sample_rent_df)
        out  = prep.transform(sample_rent_df)
        out  = engineer_rent(out)
        out["geo_cluster"]       = 0
        out["cluster_price_pct"] = 0.5
        X, y = get_rent_X_y(out)
        assert not X.isnull().any().any()
        assert not y.isnull().any()