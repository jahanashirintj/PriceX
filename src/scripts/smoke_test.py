import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directories to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from preprocess import HousePricePreprocessor
from sale_features import engineer_sale, SALE_FEATURES

def test_synchronized_pipeline():
    # 1. Training data
    df_train = pd.DataFrame({
        "area_sqft": [1000, 2000, 1500, 1200, 1800],
        "bhk": [2, 3, 2, 2, 3],
        "locality": ["A", "B", "A", "C", "A"],
        "location_type": ["URBAN", "RURAL", "URBAN", "URBAN", "RURAL"],
        "price": [5000000, 10000000, 6000000, 5500000, 9000000]
    })
    
    prep = HousePricePreprocessor(target_col="price")
    prep.fit(df_train)
    
    # 2. Prediction data (single row)
    df_pred = pd.DataFrame({
        "area_sqft": [1500],
        "bhk": [3],
        "locality": ["A"],
        "location_type": ["URBAN"],
        "age_of_property": [5],
        "furnishing": ["Semi-Furnished"]
    })
    
    # Run pipeline
    t1 = prep.transform(df_pred)
    t2 = engineer_sale(t1, stats=prep.stats_)
    
    # Check features
    print(f"Features after engineering: {t2.columns.tolist()}")
    
    for feat in SALE_FEATURES:
        if feat not in ["geo_cluster", "cluster_price_pct"]: # These need GeoLocalityClusterer
            assert feat in t2.columns, f"Missing feature: {feat}"
    
    assert t2["premium_location"].iloc[0] in [0, 1]
    assert t2["size_tier"].iloc[0] is not None
    
    print("Synchronized pipeline verified successfully!")

if __name__ == "__main__":
    test_synchronized_pipeline()
