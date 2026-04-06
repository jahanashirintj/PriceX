import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# Resolve imports from src/core
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "core"))

API_URL    = "http://localhost:8000"
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data" / "processed"

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title = "Antigravity House Price Predictor",
    page_icon  = "🏠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# --------------------------------------------------------------------------
# Premium CSS Injection
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0E1117;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #10B981 !important;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        text-align: center;
    }
    
    .stButton>button {
        border-radius: 12px;
        height: 3.5em;
        font-weight: 700;
        background: linear-gradient(to right, #3B82F6, #10B981) !important;
        border: none;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    .stSidebar {
        background-color: #161B22 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    h1, h2, h3 {
        letter-spacing: -0.02em !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_data
def load_data(listing_type: str = "sale") -> pd.DataFrame:
    path = DATA_DIR / f"{listing_type}_train.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def call_api(endpoint: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"API Connection Failed: Ensure server is running at {API_URL}"}

def fmt_price(val: float) -> str:
    if val >= 1e7: return f"₹{val/1e7:.2f} Cr"
    elif val >= 1e5: return f"₹{val/1e5:.2f} L"
    return f"₹{val:,.0f}"

# --------------------------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "🔮 Property Predictor"

with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/512/house.png", width=120)
    st.title("Price Intelligence")
    st.markdown("---")
    page = st.radio("Intelligence Hub", ["🔮 Property Predictor", "📊 Market Insights", "📈 ROI Calculator"], index=0)
    st.session_state.page = page
    st.markdown("---")
    st.info("💡 **Tip**: Enter any city and locality. Geocoding is handled automatically.")
    st.caption("AI Model: Stacked Ensemble (XGB+LGC+CAT)")

# --------------------------------------------------------------------------
# FEATURE: PREDICTOR
# --------------------------------------------------------------------------
if page == "🔮 Property Predictor":
    st.title("🔮 Market Value Predictor")
    st.markdown("Get a precision market valuation using live geocoding and state-of-the-art ML models.")

    col_input, col_viz = st.columns([1, 1.2], gap="large")

    with col_input:
        with st.form("property_form"):
            st.subheader("🏡 Property Specs")
            listing_type = st.radio("Listing Category", ["SALE", "RENT"], horizontal=True)
            
            c1, c2 = st.columns(2)
            city = c1.text_input("City", "Bangalore")
            locality = c2.text_input("Locality", "Indiranagar")
            
            area = st.number_input("Area (Sq Ft)", 300, 10000, 1500)
            bhk = st.select_slider("BHK", options=[1, 2, 3, 4, 5], value=3)
            
            f1, f2 = st.columns(2)
            age = f1.slider("Age (Years)", 0, 50, 5)
            furnish = f2.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
            
            submit = st.form_submit_button("💰 Get Valuation", use_container_width=True)

    with col_viz:
        # Check if we should run a new prediction
        if submit:
            payload = {
                "listing_type": listing_type, "area_sqft": area, "bhk": bhk,
                "location_type": "URBAN", "locality": locality, "city": city,
                "furnishing": furnish, "age_of_property": age
            }
            
            with st.spinner("🧠 Analyzing market data..."):
                res = call_api("predict", payload)
            
            if "error" in res:
                st.error(res["error"])
            else:
                # Store in session state for persistence
                st.session_state.last_prediction = res
                st.session_state.last_payload = payload
        
        # Display from session state if available
        if "last_prediction" in st.session_state:
            res = st.session_state.last_prediction
            payload = st.session_state.last_payload
            
            price = res["predicted_price"]
            lo, hi = res["confidence_interval"]
            
            # Premium Result Card
            st.markdown(f"""
            <div class="prediction-card">
                <div style='font-size: 1.1rem; opacity: 0.9; margin-bottom: 10px'>FAIR MARKET VALUE ({payload['listing_type']})</div>
                <div style='font-size: 3.8rem; font-weight: 800; line-height: 1'>{fmt_price(price)}</div>
                <div style='font-size: 0.9rem; opacity: 0.8; margin-top: 15px'>
                    Confidence Range (95%): {fmt_price(lo)} — {fmt_price(hi)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Dynamic Geolocation Map
            st.subheader("📍 Precise Location")
            
            # Use geocoded coords if available, else fallback to a general area (e.g. Bangalore center)
            lat = res.get("latitude") if res.get("geocoding_success") else 12.9716
            lng = res.get("longitude") if res.get("geocoding_success") else 77.5946
            
            if not res.get("geocoding_success"):
                st.warning(f"⚠️ Could not precisely locate '{payload['locality']}'. Mapping to {payload['city']} area.")

            m = folium.Map(location=[lat, lng], zoom_start=15, tiles="CartoDB dark_matter")
            folium.CircleMarker(
                location=[lat, lng], radius=30, color="#10B981", 
                fill=True, fill_color="#10B981", fill_opacity=0.2, popup=payload['locality']
            ).add_to(m)
            st_folium(m, height=350, use_container_width=True, key=f"map_{lat}_{lng}")

            with st.expander("🧩 Explainable AI (SHAP Factors)"):
                # Always call explain via stateful payload
                explain = call_api("explain", payload)
                if "top_factors" in explain:
                    for factor in explain["top_factors"]:
                        direction = "🟢" if factor["impact"] > 0 else "🔴"
                        st.write(f"{direction} **{factor['feature']}** — Impact: ₹{abs(factor['impact'])*1e5:,.0f}")
                elif "error" in explain:
                    st.warning(explain["error"])
        else:
            # Default state
            st.markdown("### Selection Summary")
            st.info("👈 Configure the property details to generate a high-fidelity prediction.")
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")
            st_folium(m, height=450, use_container_width=True, key="default_map")

# --------------------------------------------------------------------------
# FEATURE: INSIGHTS
# --------------------------------------------------------------------------
elif page == "📊 Market Insights":
    st.title("📊 Supply & Demand Analysis")
    df = load_data("sale")
    if not df.empty:
        tabs = st.tabs(["Price Trends", "Locality Benchmark", "Raw Data"])
        
        with tabs[0]:
            fig = px.scatter(df, x="area_sqft", y="price", color="bhk", 
                           title="Market Value vs Area (Sq Ft)", trendline="ols",
                           color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
            
        with tabs[1]:
            avg = df.groupby("locality")["price"].mean().sort_values().reset_index()
            fig = px.bar(avg, x="price", y="locality", orientation='h', color="price",
                        title="Median Market Price by Locality")
            st.plotly_chart(fig, use_container_width=True)
            
        with tabs[2]:
            st.dataframe(df.head(100), use_container_width=True)

# --------------------------------------------------------------------------
# FEATURE: ROI
# --------------------------------------------------------------------------
elif page == "📈 ROI Calculator":
    st.title("📈 Investment Intelligence")
    st.write("Determine the yield and payback period for your investment.")
    
    col_a, col_b = st.columns(2)
    sale_p = col_a.number_input("Purchase Price (₹)", value=5000000)
    rent_p = col_b.number_input("Estimated Monthly Rent (₹)", value=25000)
    
    yield_val = (rent_p * 12 / sale_p) * 100
    payback = sale_p / (rent_p * 12) if rent_p > 0 else 0
    
    m1, m2 = st.columns(2)
    m1.metric("Gross Annual Yield", f"{yield_val:.2f}%")
    m2.metric("Payback Period", f"{payback:.1f} Years")
    
    if yield_val > 4:
        st.success("✅ Strong Investment: Yield is above market average.")
    else:
        st.warning("⚠️ High Purchase Price: Consider negotiating for better yield.")