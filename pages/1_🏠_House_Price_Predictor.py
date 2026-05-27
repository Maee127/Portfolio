# pages/1_🏠_House_Price_Predictor.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CA House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ============================================
# LOAD MODELS FROM PROJECTS FOLDER
# ============================================
@st.cache_resource
def load_models():
    """Load trained model and scaler"""
    import os
    
    # Get the directory where THIS file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to Portfolio folder
    portfolio_dir = os.path.dirname(current_dir)
    
    # Build paths (works on both Windows and Linux)
    model_path = os.path.join(portfolio_dir, 'Projects', 'house_price_prediction', 'best_model.pkl')
    scaler_path = os.path.join(portfolio_dir, 'Projects', 'house_price_prediction', 'scaler.pkl')
    
    # For debugging (optional - remove after it works)
    st.write(f"Looking for model at: {model_path}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# In pages/1_🏠_House_Price_Predictor.py

@st.cache_resource
def load_models():
    """Load trained model and scaler from Projects folder"""
    # Use absolute path to where your models are
    model_path = r'C:\Users\parsayan\VS codes\ML-Portfolio\Portfolio\Projects\house_price_prediction\best_model.pkl'
    scaler_path = r'C:\Users\parsayan\VS codes\ML-Portfolio\Portfolio\Projects\house_price_prediction\scaler.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# ============================================
# SIDEBAR - MODEL INFO
# ============================================
with st.sidebar:
    st.title("📊 About This Model")
    st.markdown("---")
    
    st.markdown("### 🏆 Model Performance")
    st.metric("R² Score", "0.859", "+49% vs baseline")
    st.metric("RMSE", "$52,800", "")
    
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    st.code("""
    - Python
    - Scikit-learn
    - XGBoost
    - LightGBM
    - Streamlit
    """)
    
    st.markdown("---")
    if st.button("← Back to Portfolio"):
        st.switch_page("Home.py")

# ============================================
# MAIN CONTENT
# ============================================
st.title("🏠 California House Price Predictor")
st.markdown("*Advanced Machine Learning Ensemble (R² = 0.859)*")
st.markdown("---")

# Load models
try:
    model, scaler = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")
    st.info("Please ensure 'best_model.pkl' and 'scaler.pkl' exist in Projects/house_price_prediction/")
    st.stop()

# ============================================
# INPUT SECTION
# ============================================
st.subheader("📝 Property Details")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income ($100,000s)", 
                               min_value=0.5, max_value=15.0, value=5.0, step=0.5)
    house_age = st.slider("House Age (years)", min_value=1, max_value=52, value=20)
    avg_rooms = st.number_input("Average Rooms/Household", 
                                 min_value=2.0, max_value=10.0, value=6.0, step=0.1)
    avg_bedrooms = st.number_input("Average Bedrooms/Household", 
                                    min_value=0.5, max_value=5.0, value=1.0, step=0.1)

with col2:
    population = st.number_input("Population", min_value=100, max_value=50000, 
                                  value=2000, step=500)
    avg_occupancy = st.number_input("Average Household Occupancy", 
                                     min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    latitude = st.number_input("Latitude", min_value=32.5, max_value=42.0, 
                                value=34.0, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-124.5, max_value=-114.0, 
                                 value=-118.0, format="%.4f")

# ============================================
# PREDICTION
# ============================================
st.markdown("---")

if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
    features = np.array([[med_inc, house_age, avg_rooms, avg_bedrooms,
                          population, avg_occupancy, latitude, longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    st.balloons()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 💰 Estimated Price")
        st.markdown(f"<h1 style='text-align: center; color: #2ecc71;'>${prediction:,.0f}</h1>", 
                    unsafe_allow_html=True)

# ============================================
# MODEL RESULTS
# ============================================
st.markdown("---")
st.subheader("📊 Model Performance")

# Try to load results from CSV
results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'Projects', 'house_price_prediction', 'model_results_summary.csv')

try:
    results_df = pd.read_csv(results_path)
    st.dataframe(results_df, use_container_width=True)
except:
    st.info("Model results summary available in GitHub repository")

st.markdown("---")
st.markdown("*Model trained on California Housing dataset*")