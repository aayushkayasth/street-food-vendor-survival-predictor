"""
app.py - Complete Streamlit Application
Place this file in your Streamlit project folder along with the 'model' folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Street Food Vendor Survival Predictor",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .danger-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Load feature engineering function
def create_features(df):
    """Create advanced features for prediction"""
    df_feat = df.copy()
    
    # Business efficiency metrics
    df_feat['revenue_per_customer'] = df_feat['avg_daily_revenue_inr'] / (df_feat['avg_daily_customers'] + 1)
    df_feat['revenue_per_helper'] = df_feat['avg_daily_revenue_inr'] / (df_feat['num_helpers'] + 1)
    df_feat['customers_per_helper'] = df_feat['avg_daily_customers'] / (df_feat['num_helpers'] + 1)
    df_feat['customers_per_hour'] = df_feat['avg_daily_customers'] / (df_feat['hours_open_per_day'] + 1)
    df_feat['revenue_per_hour'] = df_feat['avg_daily_revenue_inr'] / (df_feat['hours_open_per_day'] + 1)
    
    # Experience metrics
    df_feat['age_at_start'] = df_feat['vendor_age_years'] - df_feat['years_in_business']
    df_feat['experience_ratio'] = df_feat['years_in_business'] / (df_feat['vendor_age_years'] + 1)
    
    # Competition impact
    df_feat['competition_per_customer'] = df_feat['competition_within_100m'] / (df_feat['avg_daily_customers'] + 1)
    df_feat['market_saturation'] = df_feat['competition_within_100m'] * df_feat['hours_open_per_day']
    
    # Profitability
    df_feat['monthly_revenue'] = df_feat['avg_daily_revenue_inr'] * 30
    df_feat['profit_estimate'] = df_feat['monthly_revenue'] - df_feat['monthly_stall_rent_inr']
    df_feat['profit_margin'] = df_feat['profit_estimate'] / (df_feat['monthly_revenue'] + 1)
    
    # Health and satisfaction
    df_feat['health_score_normalized'] = df_feat['monthly_health_inspection_score'] / 100
    df_feat['satisfaction_score'] = 1 - df_feat['customer_complaint_rate']
    df_feat['rent_to_revenue_ratio'] = df_feat['monthly_stall_rent_inr'] / (df_feat['monthly_revenue'] + 1)
    
    return df_feat


# Load model and components
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "deployment", "model")
# MODEL_DIR = os.path.join(BASE_DIR, "model")

@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

        with open(os.path.join(MODEL_DIR, "feature_names.json"), "r") as f:
            feature_names = json.load(f)

        with open(os.path.join(MODEL_DIR, "threshold.json"), "r") as f:
            threshold = json.load(f)["threshold"]

        return model, scaler, label_encoders, feature_names, threshold

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None
# def load_model():
#     """Load all model components"""
#     try:
#         model = joblib.load("model/xgb_model.pkl")
#         scaler = joblib.load('model/scaler.pkl')
#         label_encoders = joblib.load('model/label_encoders.pkl')
        
#         with open('model/feature_names.json', 'r') as f:
#             feature_names = json.load(f)
        
#         with open('model/optimal_threshold.json', 'r') as f:
#             threshold = json.load(f)['threshold']
        
#         return model, scaler, label_encoders, feature_names, threshold
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         st.error("Please ensure all model files are in the 'model' directory")
#         return None, None, None, None, None




# Prediction function
def predict_survival(input_data, model, scaler, label_encoders, feature_names, threshold):
    """Make prediction for single vendor"""
    
    # Create features
    input_data = create_features(input_data)
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('missing').astype(str)
            input_data[col] = input_data[col].map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
    
    # Ensure all features exist
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Select and scale features
    X = input_data[feature_names]
    X_scaled = scaler.transform(X)
    
    # Predict
    probability = model.predict_proba(X_scaled)[0][1]
    prediction = (probability >= threshold).astype(int)
    
    return prediction, probability

# Header
st.markdown("""
<div class="main-header">
    <h1>🍜 Street Food Vendor Survival Predictor</h1>
    <p>AI-powered prediction for street food business success | Trained on 12,000+ vendors</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, scaler, label_encoders, feature_names, threshold = load_model()

if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Business Information")
    st.markdown("Enter your business details below:")
    
    # Basic Information
    st.markdown("### 📍 Location & Profile")
    city = st.selectbox("City", 
        ['Delhi', 'Mumbai', 'Bengaluru', 'Hyderabad', 'Pune', 'Jaipur', 'Lucknow', 'Kochi','Surat'])
    
    zone_type = st.selectbox("Zone Type", 
        ['Commercial', 'Residential', 'Industrial', 'Transit Hub', 'University Area', 'Tourist Spot'])
    
    food_category = st.selectbox("Food Category", 
        ['Chinese', 'Chaat', 'Fast Food', 'North Indian', 'South Indian', 
         'Beverages', 'Desserts & Sweets', 'Grilled & BBQ', 'Seafood', 'Rolls & Wraps'])
    
    license_status = st.selectbox("License Status", 
        ['Licensed', 'Unlicensed', 'Expired', 'Pending Renewal'])
    
    st.markdown("### 👤 Vendor Profile")
    col1, col2 = st.columns(2)
    with col1:
        vendor_age = st.number_input("Vendor Age", min_value=18, max_value=100, value=35)
    with col2:
        years_in_business = st.number_input("Years in Business", min_value=0, max_value=50, value=5)
    
    st.markdown("### 💰 Financial Metrics")
    col1, col2 = st.columns(2)
    with col1:
        avg_revenue = st.number_input("Daily Revenue (₹)", min_value=0, max_value=50000, value=2500, step=500)
    with col2:
        avg_customers = st.number_input("Daily Customers", min_value=0, max_value=500, value=120, step=10)
    
    monthly_rent = st.number_input("Monthly Stall Rent (₹)", min_value=0, max_value=20000, value=5000, step=500)
    
    st.markdown("### 👥 Operations")
    col1, col2 = st.columns(2)
    with col1:
        num_helpers = st.number_input("Number of Helpers", min_value=0, max_value=10, value=2)
    with col2:
        hours_open = st.number_input("Hours Open Per Day", min_value=1, max_value=24, value=10)
    
    st.markdown("### 🏪 Competition & Quality")
    col1, col2 = st.columns(2)
    with col1:
        competition = st.number_input("Competitors within 100m", min_value=0, max_value=50, value=5)
    with col2:
        health_score = st.slider("Health Inspection Score", 0, 100, 80)
    
    st.markdown("### 📱 Digital Presence")
    col1, col2 = st.columns(2)
    with col1:
        had_fine = st.selectbox("Had Fine Last Year", ["No", "Yes"])
    with col2:
        online_presence = st.selectbox("Has Online Presence", ["No", "Yes"])
    
    complaint_rate = st.slider("Customer Complaint Rate", 0.0, 0.5, 0.1, 0.01, 
                                format="%.1f%%", help="Percentage of customers who complain")
    
    # Predict button
    predict_button = st.button("🔮 Predict Survival Probability", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Business Health Dashboard")
    
    # Create metrics display
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Daily Revenue", f"₹{avg_revenue:,}", 
                 delta="Good" if avg_revenue > 3000 else "Needs Improvement")
    
    with metric_col2:
        st.metric("Daily Customers", f"{avg_customers}", 
                 delta="High" if avg_customers > 150 else "Average")
    
    with metric_col3:
        profit_estimate = (avg_revenue * 30) - monthly_rent
        st.metric("Est. Monthly Profit", f"₹{profit_estimate:,}",
                 delta="Profitable" if profit_estimate > 0 else "Loss")
    
    with metric_col4:
        st.metric("Health Score", f"{health_score}/100",
                 delta="Good" if health_score >= 80 else "Needs Improvement")

with col2:
    st.markdown("### 💡 Quick Tips")
    st.info("""
    ✅ **Success Factors:**
    - Revenue > ₹3,000/day
    - Customers > 150/day
    - Health score > 80
    - Online presence
    - Low complaint rate
    
    ⚠️ **Risk Factors:**
    - High competition (>10)
    - Previous fines
    - High complaint rate (>15%)
    """)

# Make prediction when button is clicked
if predict_button:
    with st.spinner("Analyzing your business..."):
        # Prepare input data
        input_data = pd.DataFrame({
            'city': [city],
            'zone_type': [zone_type],
            'food_category': [food_category],
            'license_status': [license_status],
            'vendor_age_years': [vendor_age],
            'years_in_business': [years_in_business],
            'avg_daily_revenue_inr': [avg_revenue],
            'avg_daily_customers': [avg_customers],
            'monthly_stall_rent_inr': [monthly_rent],
            'num_helpers': [num_helpers],
            'hours_open_per_day': [hours_open],
            'competition_within_100m': [competition],
            'monthly_health_inspection_score': [health_score],
            'had_fine_last_year': [1 if had_fine == "Yes" else 0],
            'avg_monthly_rainfall_mm': [100],  # Default value
            'season_of_observation': ['Summer'],  # Default value
            'has_online_presence': [1 if online_presence == "Yes" else 0],
            'customer_complaint_rate': [complaint_rate]
        })
        
        # Make prediction
        prediction, probability = predict_survival(
            input_data, model, scaler, label_encoders, feature_names, threshold
        )
        
        # Display results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Result cards
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-card success-card">
                    <h2>✅ SURVIVES</h2>
                    <p>The model predicts your business will survive!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-card danger-card">
                    <h2>⚠️ MAY NOT SURVIVE</h2>
                    <p>The model suggests your business needs improvement.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Survival Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2ecc71" if probability > 0.5 else "#e74c3c"},
                    "steps": [
                        {"range": [0, 33], "color": "#ffcccc"},
                        {"range": [33, 66], "color": "#ffffcc"},
                        {"range": [66, 100], "color": "#ccffcc"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": threshold * 100
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with result_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence Level", f"{max(probability, 1-probability):.1%}")
            st.metric("Risk Level", "Low" if probability > 0.7 else "Medium" if probability > 0.4 else "High")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 📋 Personalized Recommendations")
        
        recommendations = []
        
        if probability < 0.4:
            recommendations.append("⚠️ **Critical Actions Needed:**")
            if avg_revenue < 3000:
                recommendations.append("• 💰 **Increase Revenue**: Consider adding popular items or increasing prices")
            if complaint_rate > 0.15:
                recommendations.append("• 📢 **Improve Quality**: High complaint rate - focus on food quality and service")
            if online_presence == "No":
                recommendations.append("• 📱 **Go Digital**: Get on food delivery apps (Zomato/Swiggy)")
            if health_score < 70:
                recommendations.append("• 🏥 **Health Compliance**: Improve hygiene to avoid fines and attract customers")
            if competition > 10:
                recommendations.append("• 🏪 **Differentiate**: High competition - offer unique items or better service")
        
        elif probability < 0.7:
            recommendations.append("📊 **Room for Improvement:**")
            if hours_open < 12:
                recommendations.append("• ⏰ **Extend Hours**: Consider opening during peak breakfast/lunch/dinner times")
            if num_helpers < 2 and avg_customers > 100:
                recommendations.append("• 👥 **Add Staff**: More helpers can improve service speed and customer satisfaction")
            if avg_revenue < 4000:
                recommendations.append("• 💡 **Boost Sales**: Try combo offers or loyalty programs")
            if complaint_rate > 0.1:
                recommendations.append("• 🎯 **Customer Service**: Respond to complaints and improve quality")
        
        else:
            recommendations.append("🌟 **Excellent Business Health!**")
            recommendations.append("• ✅ Maintain your current quality standards")
            recommendations.append("• 📈 Consider expanding to new locations")
            recommendations.append("• 🤝 Build customer loyalty programs")
            recommendations.append("• 📱 Leverage positive reviews for marketing")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Key insights
        st.markdown("### 🔍 Key Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("**Strengths:**")
            strengths = []
            if avg_revenue > 3000:
                strengths.append("✓ Strong daily revenue")
            if avg_customers > 150:
                strengths.append("✓ Good customer footfall")
            if health_score > 80:
                strengths.append("✓ Excellent health compliance")
            if online_presence == "Yes":
                strengths.append("✓ Digital presence advantage")
            if complaint_rate < 0.05:
                strengths.append("✓ Excellent customer satisfaction")
            
            if strengths:
                for s in strengths:
                    st.markdown(s)
            else:
                st.markdown("Focus on improving key metrics")
        
        with insight_col2:
            st.markdown("**Areas to Watch:**")
            weaknesses = []
            if avg_revenue < 2000:
                weaknesses.append("⚠️ Low revenue generation")
            if competition > 10:
                weaknesses.append("⚠️ High competition area")
            if complaint_rate > 0.15:
                weaknesses.append("⚠️ High complaint rate")
            if health_score < 70:
                weaknesses.append("⚠️ Health score needs improvement")
            if had_fine == "Yes":
                weaknesses.append("⚠️ Previous compliance issues")
            
            if weaknesses:
                for w in weaknesses:
                    st.markdown(w)
            else:
                st.markdown("✓ No major concerns identified")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>Powered by XGBoost | Trained on 12,000+ vendors | Accuracy: 85%</p>
    <p>This tool provides predictions based on historical data. Results are for guidance only.</p>
</div>
""", unsafe_allow_html=True)