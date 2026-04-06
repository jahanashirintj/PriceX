"""
geo_cluster.py
==============
K-Means geo-locality clustering.

Groups localities by price behaviour AND physical location.
Adds two features:
  - geo_cluster       : cluster ID (int)
  - cluster_price_pct : within-cluster price rank 0.0–1.0
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class GeoLocalityClusterer:
    """
    Unsupervised geo-clustering using K-Means on
    [latitude, longitude, locality_price_idx].

    Parameters
    ----------
    n_clusters   : Number of geo-price clusters (default 12).
    random_state : Reproducibility seed.
    """

    def __init__(self, n_clusters: int = 12, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model  = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            max_iter=300,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def _get_coords(self, df: pd.DataFrame) -> np.ndarray:
        cols = ["latitude", "longitude", "locality_price_idx"]
        return df[cols].fillna(0).values

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "GeoLocalityClusterer":
        """Fit on training data. Requires latitude, longitude, locality_price_idx."""
        coords = self._get_coords(df)
        scaled = self._scaler.fit_transform(coords)
        self._model.fit(scaled)
        self._fitted = True
        print(f"[GeoClusterer] Fitted {self.n_clusters} clusters "
              f"on {len(df):,} rows.")
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign cluster IDs and within-cluster price rank."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        coords = self._get_coords(df)
        scaled = self._scaler.transform(coords)
        df["geo_cluster"] = self._model.predict(scaled)

        # Rank each row's locality_price_idx within its cluster (0–1)
        df["cluster_price_pct"] = df.groupby("geo_cluster")[
            "locality_price_idx"
        ].rank(pct=True)

        return df

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    def cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-cluster price statistics for analysis."""
        df = self.transform(df)
        return (
            df.groupby("geo_cluster")
            .agg(
                count=("price", "count"),
                mean_price=("price", "mean"),
                median_price=("price", "median"),
                std_price=("price", "std"),
            )
            .reset_index()
            .sort_values("mean_price", ascending=False)
        )

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"[GeoClusterer] Saved to {path}")

    @staticmethod
    def load(path: str) -> "GeoLocalityClusterer":
        obj = joblib.load(path)
        print(f"[GeoClusterer] Loaded from {path}")
        return obj