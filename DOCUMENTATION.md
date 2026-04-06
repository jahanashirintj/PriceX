# House Price Prediction System Documentation

## Overview
The **Antigravity House Price Predictor** is a high-fidelity machine learning system designed to estimate the market value of residential properties for both sale and rent. It leverages a state-of-the-art stacked ensemble architecture and provides a dual-layer interface: a high-performance FastAPI backend and a premium Streamlit visualization dashboard.

---

## 🏗️ System Architecture

### 1. ML Model (Stacked Ensemble)
The core prediction engine uses a two-layer stacking approach:
- **Base Learners (Level 0)**:
    - **XGBoost**: Captures complex non-linear interactions.
    - **LightGBM**: Efficient gradient boosting for large-scale data.
    - **CatBoost**: Excels at handling categorical features (e.g., location types).
- **Meta-Learner (Level 1)**:
    - **Ridge Regression**: Learns the optimal weighted combination of the base learners' predictions to minimize overall error and variance.
- **Passthrough**: The meta-learner is also trained on the original feature set, allowing it to "correct" base learner errors using raw data.

### 2. Feature Engineering
The system applies advanced domain-specific features:
- **Geospatial Clustering**: Groups properties into localized price clusters using latitude/longitude.
- **ROI Metrics**: Calculates annual yield (%) and breakeven years for investment analysis.
- **Socio-Economic Indicators**: Maps urban vs. rural price variances and locality price indices.

---

## 🚀 API Reference (FastAPI)

The backend service runs on port `8000` by default.

### Endpoints

#### `POST /predict`
Predicts the price for a single property.
- **Payload**: `PropertyInput` (listing_type, area_sqft, bhk, city, locality, etc.)
- **Response**: `PredictionResponse` (predicted_price, confidence_interval, geocoding_success).

#### `POST /explain`
Provides Explainable AI (XAI) insights into the prediction.
- **Response**: Top 5 SHAP (Shapley Additive Explanations) factors contributing to the price.

#### `POST /compare`
Generates an ROI comparison between buying and renting the same property.
- **Response**: `CompareResponse` (annual_yield_pct, breakeven_years, recommendation).

#### `GET /health`
Liveness check for the API and model loading status.

---

## 📊 Dashboard Guide (Streamlit)

The frontend interface runs on port `8501`.

### Sections
1.  **🔮 Property Predictor**: Input property specifications to receive an instant valuation and an interactive Folium map showing the property's precise location.
2.  **📊 Market Insights**: Visualizes supply and demand trends, price-to-area scatter plots, and locality benchmarks.
3.  **📈 ROI Calculator**: A tool for investors to calculate gross annual yield and payback periods manually or via API comparison.

---

## 📁 Project Structure

- `src/api/main.py`: FastAPI server and endpoint definitions.
- `src/dashboard/app.py`: Streamlit dashboard and UI components.
- `src/core/`: Contains the underlying logic.
    - `stacker.py`: Ensemble training and persistence.
    - `preprocess.py`: Data cleaning and scaling.
    - `geo_cluster.py`: Geospatial clustering logic.
- `models/`: Persisted `.pkl` files for models and preprocessors.
- `data/`: CSV datasets for training and EDA.
- `tests/`: Unit and integration testing suite.

---

## ⚙️ Dependencies
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`, `mlflow`
- **Geospatial**: `geopy`, `folium`, `streamlit-folium`
- **Backend/Frontend**: `fastapi`, `uvicorn`, `streamlit`, `plotly`
