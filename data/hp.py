# data_cleaning_and_split.py

import pandas as pd
import numpy as np

def clean_and_split_data(path):
    # Load data
    df = pd.read_csv("raw.csv")

    print("Original Shape:", df.shape)

    # -------------------------------
    # 1. Standardize column names
    # -------------------------------
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # -------------------------------
    # 2. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 3. Handle missing values
    # -------------------------------
    df = df.dropna(subset=["price", "listing_type"])

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # -------------------------------
    # 4. Fix types
    # -------------------------------
    df["bhk"] = df["bhk"].astype(int)
    df["area_sqft"] = df["area_sqft"].astype(float)
    df["price"] = df["price"].astype(float)

    # -------------------------------
    # 5. Clean categorical values
    # -------------------------------
    df["listing_type"] = df["listing_type"].str.upper().str.strip()
    df["location_type"] = df["location_type"].str.upper().str.strip()
    df["furnishing"] = df["furnishing"].str.title().str.strip()

    # -------------------------------
    # 6. Remove invalid values
    # -------------------------------
    df = df[(df["price"] > 0) & (df["area_sqft"] > 0) & (df["bhk"] > 0)]

    # -------------------------------
    # 7. Remove outliers
    # -------------------------------
    def remove_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]

    df = remove_outliers(df, "price")
    df = remove_outliers(df, "area_sqft")

    # -------------------------------
    # 8. Feature engineering
    # -------------------------------
    df["price_per_sqft"] = df["price"] / df["area_sqft"]
    df["log_price"] = np.log1p(df["price"])

    print("Cleaned Shape:", df.shape)

    # -------------------------------
    # 9. 🔥 SPLIT INTO SALE & RENT
    # -------------------------------
    sale_df = df[df["listing_type"] == "SALE"].copy()
    rent_df = df[df["listing_type"] == "RENT"].copy()

    print("SALE data:", sale_df.shape)
    print("RENT data:", rent_df.shape)

    return sale_df, rent_df


if __name__ == "__main__":
    sale_df, rent_df = clean_and_split_data("data/raw/house_data.csv")

    # Save separately
    sale_df.to_csv("data/processed/sale_data.csv", index=False)
    rent_df.to_csv("data/processed/rent_data.csv", index=False)

    print("Saved sale_data.csv and rent_data.csv")