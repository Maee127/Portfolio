# Home.py - Portfolio Homepage
import streamlit as st

st.set_page_config(
    page_title="Data Science Portfolio",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 My Data Science Portfolio")
st.markdown("## *Machine Learning • Predictive Modeling • Data Analysis*")
st.markdown("---")

# About me
st.markdown("### 👋 About Me")
st.markdown("""
Data Scientist passionate about building predictive models that solve real-world problems.
- **R² Score Achieved**: 0.859 on California Housing
- **47% improvement** over baseline linear models
- **Stacking Ensemble** with XGBoost, LightGBM, and Gradient Boosting
""")

st.markdown("---")
st.markdown("### 📁 Featured Projects")

# Project 1
st.markdown("#### 🏠 California House Price Predictor")
st.markdown("""
**R² Score: 0.859** | **RMSE: $52,800**

Advanced ensemble model combining XGBoost, LightGBM, and Gradient Boosting 
with stacking architecture.

👉 **Click the link in the sidebar to try the live predictor!**
""")

st.markdown("---")
st.markdown("*Built with Streamlit • Deployed on Streamlit Community Cloud*")