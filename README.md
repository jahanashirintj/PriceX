# 🏠 Antigravity House Price Prediction System

**High-Fidelity Machine Learning Solution for Real Estate Valuation**

The **Antigravity House Price Predictor** is a sophisticated end-to-end system designed to provide precise market valuations for residential properties. It focuses on delivering actionable intelligence for both the sale and rental markets, addressing the inherent complexity of non-linear property features through advanced ensemble learning.

### 🤖 Machine Learning Core
The project implements a **Stacked Ensemble Learning** architecture to achieve superior accuracy over standard single-model baselines.
- **Layer 0 (Base Learners)**: A combination of **XGBoost**, **LightGBM**, and **CatBoost** is used to capture varied structural and geographic patterns from the data.
- **Layer 1 (Meta-Learner)**: A **Ridge Regression** model serves as the meta-learner, learning the optimal weights for the base learners’ outputs.
- **Explainable AI (SHAP)**: Every prediction is deconstructed to show exactly which features (such as location clusters, house age, or floor area) influenced the final price.

### 📍 Key Capabilities
- **Geospatial Clustering**: Employs K-Means clustering on latitude, longitude, and historical price indices to identify localized micro-markets and economic zones.
- **Investment ROI Engine**: Automatically compares sale vs. rental predictions to calculate gross annual yields and investment payback periods for a single property.
- **Robust Preprocessing**: Stateful pipelines including smoothed target encoding for localities, IQR-based outlier clipping, and automated geocoding via Nominatim (Geopy).

### 🚀 Tech Stack
- **Backend API**: High-performance **FastAPI** for RESTful serving and model inference.
- **Frontend UI**: Premium **Streamlit** dashboard with interactive **Plotly** analytics and **Folium** mapping.
- **Project Governance**: Comprehensive unit tests, automated setup scripts (`setup.bat` and `RUN.bat`), and experiment tracking via **MLflow**.

---
*For a deep-dive into the methodology and API reference, please refer to [DOCUMENTATION.md](DOCUMENTATION.md).*
