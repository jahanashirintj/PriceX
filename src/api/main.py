"""
main.py  (FastAPI)
==================
REST API serving layer for the house price prediction system.

Endpoints:
  POST /predict  — single or batch price prediction
  GET  /explain  — SHAP top-5 feature explanations
  GET  /compare  — sale vs rent ROI comparison
  GET  /health   — liveness check
  GET  /features — list of features used by each model

Run:
    uvicorn src.api.main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path
from typing import Literal, List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Resolve imports from src/core
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "core"))

import ssl
import certifi
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from evaluate import get_top_shap_factors
from sale_features import engineer_sale
from rent_features import engineer_rent
from geo_cluster import GeoLocalityClusterer


# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = FastAPI(
    title       = "House Price Prediction API",
    description = "Stacked ensemble (XGBoost + LightGBM + CatBoost + Ridge meta)",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# --------------------------------------------------------------------------
# Model registry (loaded at startup)
# --------------------------------------------------------------------------
MODELS_DIR = ROOT / "models"

_sale_model   = None
_rent_model   = None
_sale_prep    = None
_rent_prep    = None
_geo_cluster  = None


@app.on_event("startup")
def load_models():
    global _sale_model, _rent_model, _sale_prep, _rent_prep, _geo_cluster
    try:
        _sale_model  = joblib.load(MODELS_DIR / "sale_model.pkl")
        _rent_model  = joblib.load(MODELS_DIR / "rent_model.pkl")
        _sale_prep   = joblib.load(MODELS_DIR / "sale_preprocessor.pkl")
        _rent_prep   = joblib.load(MODELS_DIR / "rent_preprocessor.pkl")
        
        if (MODELS_DIR / "geo_clusterer.pkl").exists():
            _geo_cluster = GeoLocalityClusterer.load(MODELS_DIR / "geo_clusterer.pkl")
        
        print("[API] Models and clusterer loaded successfully.")
    except FileNotFoundError as e:
        print(f"[API] Warning — model file not found: {e}")
        print("[API] Train models first: python train_models.py")


# Feature lists (Definitive Source of Truth)
SALE_FEATURES = [
    "area_sqft", "bhk", "is_urban", "bhk_area_ratio", "urban_x_area", "bhk_squared", "age_x_bhk",
    "log_area", "locality_price_idx", "log_locality_idx", "geo_cluster", "cluster_price_pct",
    "age_of_property", "size_tier", "premium_location", "value_density"
]

RENT_FEATURES = [
    "area_sqft", "bhk", "is_urban", "bhk_area_ratio", "urban_x_area", "bhk_squared", "age_x_bhk",
    "log_area", "locality_price_idx", "log_locality_idx", "geo_cluster", "cluster_price_pct",
    "furnished_score", "is_furnished", "is_semi_furnished", "bhk_x_furnish", "monthly_yield",
    "furnish_x_urban", "small_unit_flag", "age_of_property"
]


# --------------------------------------------------------------------------
# Request / Response schemas
# --------------------------------------------------------------------------
class PropertyInput(BaseModel):
    listing_type:    Literal["SALE", "RENT"]
    area_sqft:       float   = Field(..., gt=0, le=10000, description="Area in sqft")
    bhk:             int     = Field(..., ge=1, le=10)
    location_type:   Literal["URBAN", "RURAL"]
    locality:        str     = Field(..., min_length=1)
    furnishing:      Literal["Unfurnished", "Semi-Furnished", "Furnished"] = "Unfurnished"
    age_of_property: int     = Field(default=5, ge=0, le=100)
    city:            str     = Field(default="Bangalore", min_length=2)
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None


class BatchInput(BaseModel):
    properties: List[PropertyInput]


class PredictionResponse(BaseModel):
    predicted_price:      float
    confidence_interval:  List[float]
    listing_type:         str
    area_sqft:            float
    bhk:                  int
    latitude:             Optional[float] = None
    longitude:            Optional[float] = None
    geocoding_success:    bool = False


class ExplainResponse(BaseModel):
    predicted_price: float
    top_factors:     List[dict]


class CompareResponse(BaseModel):
    sale_price:        float
    monthly_rent:      float
    annual_yield_pct:  float
    breakeven_years:   float
    recommendation:    str


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _get_coordinates(locality: str, city: str):
    """Fetch lat/lng using Nominatim API with SSL fix."""
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
        geolocator = Nominatim(
            user_agent="house_price_predictor_v2", 
            ssl_context=ctx,
            timeout=10
        )
        location = geolocator.geocode(f"{locality}, {city}")
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, Exception):
        pass
    return None, None


def _preprocess_and_predict(prop: PropertyInput) -> tuple:
    """Returns (predicted_price, features_df, model)."""
    is_sale = prop.listing_type == "SALE"
    prep    = _sale_prep if is_sale else _rent_prep
    model   = _sale_model if is_sale else _rent_model
    features = SALE_FEATURES if is_sale else RENT_FEATURES
    engineer = engineer_sale if is_sale else engineer_rent

    if model is None or prep is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Train models first."
        )

    # 1. Real-world Geocoding if coordinates missing
    lat, lng = prop.latitude, prop.longitude
    if lat is None or lng is None:
        lat, lng = _get_coordinates(prop.locality, prop.city)

    # 2. Base Preprocessing
    raw = pd.DataFrame([prop.dict()])
    df  = prep.transform(raw)

    # 3. Geo-Clustering (if clusterer available and coords found)
    if _geo_cluster and lat is not None and lng is not None:
        # Update row with geocoded status for clustering
        df["latitude"]  = lat
        df["longitude"] = lng
        df = _geo_cluster.transform(df)
    else:
        # Fallback to placeholders if geocoding failed
        if "geo_cluster" not in df.columns:
            df["geo_cluster"]       = 0
            df["cluster_price_pct"] = 0.5
    
    # 4. Specialized Feature Engineering
    df = engineer(df, stats=getattr(prep, "stats_", None))

    # 5. Final Feature Alignment
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]
    log_pred = float(model.predict(X)[0])
    price    = float(np.expm1(log_pred))

    return price, X, model, lat, lng


def _confidence_interval(price: float, pct: float = 0.08) -> List[float]:
    """Simple ±pct confidence band."""
    return [round(price * (1 - pct), 2), round(price * (1 + pct), 2)]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "sale_model":   _sale_model is not None,
        "rent_model":   _rent_model is not None,
    }


@app.get("/features")
def list_features():
    return {
        "SALE": SALE_FEATURES,
        "RENT": RENT_FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(prop: PropertyInput):
    """
    Predict price for a single property.
    Returns predicted price and 8% confidence interval.
    """
    price, _, _, lat, lng = _preprocess_and_predict(prop)
    return PredictionResponse(
        predicted_price     = round(price, 2),
        confidence_interval = _confidence_interval(price),
        listing_type        = prop.listing_type,
        area_sqft           = prop.area_sqft,
        bhk                 = prop.bhk,
        latitude            = lat,
        longitude           = lng,
        geocoding_success   = lat is not None and lng is not None
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    """Predict prices for a list of properties."""
    results = []
    for prop in batch.properties:
        try:
            price, _, _, _, _ = _preprocess_and_predict(prop)
            results.append({
                "predicted_price":     round(price, 2),
                "confidence_interval": _confidence_interval(price),
                "listing_type":        prop.listing_type,
                "bhk":                 prop.bhk,
                "area_sqft":           prop.area_sqft,
            })
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}


@app.post("/explain", response_model=ExplainResponse)
def explain(prop: PropertyInput):
    """
    Predict price and return top-5 SHAP feature contributions.
    """
    price, X, model, _, _ = _preprocess_and_predict(prop)
    top_factors = get_top_shap_factors(model, X, n=5)
    return ExplainResponse(
        predicted_price = round(price, 2),
        top_factors     = top_factors,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(prop: PropertyInput):
    """
    Compare sale price vs rental income for the same property.
    Returns ROI metrics and a buy/rent recommendation.
    """
    # Get sale price
    sale_prop         = prop.dict().copy()
    sale_prop["listing_type"] = "SALE"
    rent_prop         = prop.dict().copy()
    rent_prop["listing_type"] = "RENT"
    
    sale_price, _, _, _, _  = _preprocess_and_predict(PropertyInput(**sale_prop))
    monthly_rent, _, _, _, _ = _preprocess_and_predict(PropertyInput(**rent_prop))

    annual_rent      = monthly_rent * 12
    annual_yield_pct = (annual_rent / sale_price) * 100
    breakeven_years  = sale_price / annual_rent if annual_rent > 0 else 999

    if annual_yield_pct >= 5:
        rec = "Buy — strong rental yield above 5%"
    elif breakeven_years <= 20:
        rec = f"Buy — breakeven in {breakeven_years:.1f} years"
    else:
        rec = "Rent — purchase price too high relative to rental income"

    return CompareResponse(
        sale_price       = round(sale_price, 2),
        monthly_rent     = round(monthly_rent, 2),
        annual_yield_pct = round(annual_yield_pct, 2),
        breakeven_years  = round(breakeven_years, 1),
        recommendation   = rec,
    )


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)