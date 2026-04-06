"""
feature_store.py
================
Builds, validates, and persists the processed feature sets for
both sale and rent pipelines.

Usage:
    python feature_store.py --raw data/raw/house_data.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "core"))

from preprocess import HousePricePreprocessor
from sale_features import engineer_sale, SALE_FEATURES, get_sale_X_y
from rent_features import engineer_rent, RENT_FEATURES, get_rent_X_y
from geo_cluster import GeoLocalityClusterer


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------
REQUIRED_COLS = [
    "price", "area_sqft", "bhk", "listing_type",
    "location_type", "locality",
]

OPTIONAL_COLS = [
    "latitude", "longitude", "furnishing",
    "age_of_property", "amenities",
]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[feature_store] Missing required columns: {missing}")
    print(f"[feature_store] Schema OK — {df.shape[0]:,} rows, {df.shape[1]} cols")


# --------------------------------------------------------------------------
# Split helpers
# --------------------------------------------------------------------------
def split_listing(df: pd.DataFrame):
    df_sale = df[df["listing_type"] == "SALE"].copy().reset_index(drop=True)
    df_rent = df[df["listing_type"] == "RENT"].copy().reset_index(drop=True)
    print(f"[feature_store] Sale rows: {len(df_sale):,} | Rent rows: {len(df_rent):,}")
    return df_sale, df_rent


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def build_features(
    raw_path: str,
    out_dir: str = "models",
    processed_dir: str = "data/processed",
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load raw data
    print(f"\n[feature_store] Loading {raw_path}")
    df = pd.read_csv(raw_path)
    validate_schema(df)

    # 2. Split listing types
    df_sale, df_rent = split_listing(df)

    # 3. Train/test split (time-based if date col exists, else random)
    def ttsplit(data):
        n = int(len(data) * train_ratio)
        idx = data.sample(frac=1, random_state=random_state).index
        return data.loc[idx[:n]], data.loc[idx[n:]]

    sale_train, sale_test = ttsplit(df_sale)
    rent_train, rent_test = ttsplit(df_rent)

    # 4. Fit preprocessors (on train only)
    sale_prep = HousePricePreprocessor(log_transform=True)
    rent_prep = HousePricePreprocessor(log_transform=True)
    sale_prep.fit(sale_train)
    rent_prep.fit(rent_train)

    # 5. Transform all splits
    sale_train_t = sale_prep.transform(sale_train)
    sale_test_t  = sale_prep.transform(sale_test)
    rent_train_t = rent_prep.transform(rent_train)
    rent_test_t  = rent_prep.transform(rent_test)

    # 6. Geo clustering (fit on train, transform all)
    if "latitude" in df.columns and "longitude" in df.columns:
        geo = GeoLocalityClusterer(n_clusters=12)
        geo.fit(sale_train_t)
        sale_train_t = geo.transform(sale_train_t)
        sale_test_t  = geo.transform(sale_test_t)
        rent_train_t = geo.transform(rent_train_t)
        rent_test_t  = geo.transform(rent_test_t)
        joblib.dump(geo, os.path.join(out_dir, "geo_clusterer.pkl"))
        print("[feature_store] Geo clustering applied.")
    else:
        # Placeholder columns so downstream code doesn't break
        for d in [sale_train_t, sale_test_t, rent_train_t, rent_test_t]:
            d["geo_cluster"]       = 0
            d["cluster_price_pct"] = 0.5
        print("[feature_store] No lat/lng — geo_cluster set to 0.")

    # 7. Listing-specific feature engineering (pass preprocessor stats)
    sale_train_t = engineer_sale(sale_train_t, stats=sale_prep.stats_)
    sale_test_t  = engineer_sale(sale_test_t,  stats=sale_prep.stats_)
    rent_train_t = engineer_rent(rent_train_t, stats=rent_prep.stats_)
    rent_test_t  = engineer_rent(rent_test_t,  stats=rent_prep.stats_)

    # 8. Persist processed CSVs
    for name, data in [
        ("sale_train", sale_train_t),
        ("sale_test",  sale_test_t),
        ("rent_train", rent_train_t),
        ("rent_test",  rent_test_t),
    ]:
        path = os.path.join(processed_dir, f"{name}.csv")
        data.to_csv(path, index=False)
        print(f"[feature_store] Saved {path}  shape={data.shape}")

    # 9. Persist preprocessors
    joblib.dump(sale_prep, os.path.join(out_dir, "sale_preprocessor.pkl"))
    joblib.dump(rent_prep, os.path.join(out_dir, "rent_preprocessor.pkl"))
    print(f"[feature_store] Preprocessors saved to {out_dir}/")

    # 10. Print feature summary
    print("\n[feature_store] Sale features:", SALE_FEATURES)
    print("[feature_store] Rent features:", RENT_FEATURES)
    print("\n[feature_store] Done.\n")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature store")
    parser.add_argument("--raw",       default="data/raw/house_data.csv")
    parser.add_argument("--out_dir",   default="models")
    parser.add_argument("--proc_dir",  default="data/processed")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    build_features(
        raw_path=args.raw,
        out_dir=args.out_dir,
        processed_dir=args.proc_dir,
        train_ratio=args.train_ratio,
    )