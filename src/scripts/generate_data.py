import pandas as pd
import numpy as np
import os
import argparse

def generate_realistic_data(n=2000, out_path="data/raw_synthetic.csv"):
    """
    Generates synthetic house price data with realistic correlations 
    to fix the 'same amount' prediction issue.
    """
    np.random.seed(42)
    
    # 1. Base features
    area = np.random.randint(400, 5000, n)
    bhk  = np.array([1 if a < 800 else (2 if a < 1500 else (3 if a < 2500 else (4 if a < 3500 else 5))) for a in area])
    # Add some randomness to bhk
    bhk = np.clip(bhk + np.random.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1]), 1, 5)
    
    localities = ["Koramangala", "Whitefield", "HSR Layout", "Indiranagar", "Electronic City", "Bellandur"]
    locality = np.random.choice(localities, n)
    loc_premium = {
        "Koramangala": 1.5, "Indiranagar": 1.4, "HSR Layout": 1.2, 
        "Bellandur": 1.1, "Whitefield": 1.0, "Electronic City": 0.8
    }
    
    location_type = np.random.choice(["URBAN", "RURAL"], n, p=[0.8, 0.2])
    urban_mult = np.where(location_type == "URBAN", 1.2, 0.8)
    
    age = np.random.randint(0, 30, n)
    age_mult = 1 - (age * 0.01) # 1% depreciation per year
    
    furnishing = np.random.choice(["Unfurnished", "Semi-Furnished", "Furnished"], n)
    furn_mult = {"Unfurnished": 1.0, "Semi-Furnished": 1.1, "Furnished": 1.25}
    
    # 2. Target Variable (Price)
    # Price = (Base + Area*Rate + BHK*Bonus) * Locality * Urban * Age * Furn + Noise
    base_price = 1000000
    sqft_rate  = 4500
    bhk_bonus  = 600000
    
    prices = (base_price + area * sqft_rate + bhk * bhk_bonus)
    prices = prices * np.array([loc_premium[l] for l in locality])
    prices = prices * urban_mult * age_mult * np.array([furn_mult[f] for f in furnishing])
    
    # Add 5% noise
    noise = np.random.normal(1.0, 0.05, n)
    prices = prices * noise
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        "area_sqft": area,
        "bhk": bhk,
        "locality": locality,
        "location_type": location_type,
        "age_of_property": age,
        "furnishing": furnishing,
        "latitude": np.random.uniform(12.8, 13.1, n), # Bangalore range
        "longitude": np.random.uniform(77.4, 77.8, n),
        "price": prices.astype(int)
    })
    
    # Randomly split into SALE and RENT for the demo
    df["listing_type"] = np.random.choice(["SALE", "RENT"], n)
    # Adjust prices for rent (approx 3.5% annual yield)
    df.loc[df["listing_type"] == "RENT", "price"] = (df.loc[df["listing_type"] == "RENT", "price"] * 0.035 / 12).astype(int)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[generate_data] Success: {n} realistic records saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--out", type=str, default="data/raw_synthetic.csv")
    args = parser.parse_args()
    generate_realistic_data(args.n, args.out)
