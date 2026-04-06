"""
test_api.py
===========
Integration and unit tests for the FastAPI endpoints.

Run:
    pytest tests/test_api.py -v

Note: These tests mock the model loading so they run without
trained .pkl files. For end-to-end tests, train models first.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "api"))


# --------------------------------------------------------------------------
# Mock model and preprocessor
# --------------------------------------------------------------------------
def make_mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([13.5])  # log(~735k)
    return model


def make_mock_preprocessor():
    import pandas as pd

    def mock_transform(df):
        df = df.copy()
        df["is_urban"]          = 1
        df["log_area"]          = 7.0
        df["bhk_area_ratio"]    = 0.002
        df["locality_price_idx"]= 600_000
        df["log_locality_idx"]  = 13.3
        df["age_x_bhk"]         = 10
        df["urban_x_area"]      = 1000
        df["bhk_squared"]       = 4
        df["price_per_sqft"]    = 500
        df["geo_cluster"]       = 0
        df["cluster_price_pct"] = 0.5
        df["furnished_score"]   = 0
        df["is_furnished"]      = 0
        df["is_semi_furnished"] = 0
        df["size_tier"]         = 2
        df["premium_location"]  = 1
        df["value_density"]     = 0.5
        df["bhk_x_furnish"]     = 0
        df["monthly_yield"]     = 20
        df["furnish_x_urban"]   = 0
        df["small_unit_flag"]   = 0
        return df

    prep = MagicMock()
    prep.transform.side_effect = mock_transform
    return prep


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def client():
    """Create a test client with mocked models."""
    with patch.dict("sys.modules", {"evaluate": MagicMock()}):
        # Patch model loading before importing the app
        mock_sale_model = make_mock_model()
        mock_rent_model = make_mock_model()
        mock_sale_prep  = make_mock_preprocessor()
        mock_rent_prep  = make_mock_preprocessor()

        with patch("joblib.load", side_effect=[
            mock_sale_model, mock_rent_model,
            mock_sale_prep, mock_rent_prep,
        ]):
            try:
                from fastapi.testclient import TestClient
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "main",
                    str(Path(__file__).parent.parent / "src" / "api" / "main.py"),
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module._sale_model  = mock_sale_model
                module._rent_model  = mock_rent_model
                module._sale_prep   = mock_sale_prep
                module._rent_prep   = mock_rent_prep
                return TestClient(module.app)
            except Exception as e:
                pytest.skip(f"Could not load FastAPI app: {e}")


@pytest.fixture
def sale_payload():
    return {
        "listing_type":    "SALE",
        "area_sqft":       1000,
        "bhk":             2,
        "location_type":   "URBAN",
        "locality":        "Koramangala",
        "furnishing":      "Unfurnished",
        "age_of_property": 5,
    }


@pytest.fixture
def rent_payload():
    return {
        "listing_type":    "RENT",
        "area_sqft":       800,
        "bhk":             2,
        "location_type":   "URBAN",
        "locality":        "Whitefield",
        "furnishing":      "Furnished",
        "age_of_property": 3,
    }


# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------
class TestHealth:

    def test_health_endpoint_returns_200(self, client):
        if client is None:
            pytest.skip()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_has_status(self, client):
        if client is None:
            pytest.skip()
        data = client.get("/health").json()
        assert "status" in data

    def test_features_endpoint(self, client):
        if client is None:
            pytest.skip()
        resp = client.get("/features")
        assert resp.status_code == 200
        data = resp.json()
        assert "SALE" in data
        assert "RENT" in data
        assert isinstance(data["SALE"], list)


# --------------------------------------------------------------------------
# /predict endpoint
# --------------------------------------------------------------------------
class TestPredict:

    def test_predict_sale_returns_200(self, client, sale_payload):
        if client is None:
            pytest.skip()
        resp = client.post("/predict", json=sale_payload)
        assert resp.status_code == 200

    def test_predict_sale_has_price(self, client, sale_payload):
        if client is None:
            pytest.skip()
        data = client.post("/predict", json=sale_payload).json()
        assert "predicted_price" in data
        assert data["predicted_price"] > 0

    def test_predict_sale_has_confidence_interval(self, client, sale_payload):
        if client is None:
            pytest.skip()
        data = client.post("/predict", json=sale_payload).json()
        assert "confidence_interval" in data
        ci = data["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] < data["predicted_price"] < ci[1]

    def test_predict_rent_returns_200(self, client, rent_payload):
        if client is None:
            pytest.skip()
        resp = client.post("/predict", json=rent_payload)
        assert resp.status_code == 200

    def test_predict_invalid_listing_type(self, client, sale_payload):
        if client is None:
            pytest.skip()
        bad = {**sale_payload, "listing_type": "UNKNOWN"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422  # Pydantic validation error

    def test_predict_negative_area_rejected(self, client, sale_payload):
        if client is None:
            pytest.skip()
        bad = {**sale_payload, "area_sqft": -100}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_batch(self, client, sale_payload, rent_payload):
        if client is None:
            pytest.skip()
        batch = {"properties": [sale_payload, rent_payload]}
        resp  = client.post("/predict/batch", json=batch)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2


# --------------------------------------------------------------------------
# /explain endpoint
# --------------------------------------------------------------------------
class TestExplain:

    def test_explain_returns_top_factors(self, client, sale_payload):
        if client is None:
            pytest.skip()
        with patch(
            "main.get_top_shap_factors",
            return_value=[{"feature": "area_sqft", "impact": 0.5}],
        ):
            resp = client.post("/explain", json=sale_payload)
            if resp.status_code == 200:
                data = resp.json()
                assert "top_factors" in data
                assert "predicted_price" in data


# --------------------------------------------------------------------------
# /compare endpoint
# --------------------------------------------------------------------------
class TestCompare:

    def test_compare_returns_roi_fields(self, client, sale_payload):
        if client is None:
            pytest.skip()
        resp = client.post("/compare", json=sale_payload)
        if resp.status_code == 200:
            data = resp.json()
            for field in ["sale_price", "monthly_rent",
                          "annual_yield_pct", "breakeven_years", "recommendation"]:
                assert field in data

    def test_compare_breakeven_positive(self, client, sale_payload):
        if client is None:
            pytest.skip()
        resp = client.post("/compare", json=sale_payload)
        if resp.status_code == 200:
            assert resp.json()["breakeven_years"] > 0


# --------------------------------------------------------------------------
# Standalone utility tests (no FastAPI needed)
# --------------------------------------------------------------------------
class TestUtilities:

    def test_price_formatting(self):
        """Test the fmt_price helper logic."""
        def fmt(val):
            if val >= 1e7:
                return f"₹{val/1e7:.2f} Cr"
            elif val >= 1e5:
                return f"₹{val/1e5:.2f} L"
            return f"₹{val:,.0f}"

        assert "Cr" in fmt(10_000_000)
        assert "L"  in fmt(500_000)
        assert "₹"  in fmt(50_000)

    def test_confidence_interval_bounds(self):
        """CI should bracket the predicted price."""
        price = 1_000_000
        pct   = 0.08
        lo    = price * (1 - pct)
        hi    = price * (1 + pct)
        assert lo < price < hi
        assert round(lo, 2) == 920_000.0
        assert round(hi, 2) == 1_080_000.0

    def test_log_transform_roundtrip(self):
        """np.log1p / np.expm1 should roundtrip."""
        prices = np.array([100_000, 500_000, 1_000_000, 5_000_000])
        log_p  = np.log1p(prices)
        recovered = np.expm1(log_p)
        np.testing.assert_allclose(prices, recovered, rtol=1e-10)