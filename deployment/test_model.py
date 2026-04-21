import joblib
import json
import pandas as pd
from feature_engineering import create_features

# Load
model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

with open("model/feature_names.json") as f:
    feature_names = json.load(f)

# Sample input
data = pd.DataFrame({
    'city': ['Delhi'],
    'zone_type': ['Commercial'],
    'food_category': ['Fast Food'],
    'license_status': ['Licensed'],
    'vendor_age_years': [35],
    'years_in_business': [5],
    'avg_daily_revenue_inr': [2500],
    'avg_daily_customers': [120],
    'monthly_stall_rent_inr': [5000],
    'num_helpers': [2],
    'hours_open_per_day': [10],
    'competition_within_100m': [3],
    'monthly_health_inspection_score': [85],
    'had_fine_last_year': [0],
    'avg_monthly_rainfall_mm': [100],
    'season_of_observation': ['Summer'],
    'has_online_presence': [1],
    'customer_complaint_rate': [0.05]
})

# Feature engineering
data = create_features(data)

# Encode
for col, encoder in label_encoders.items():
    if col in data.columns:
        data[col] = encoder.transform(data[col])

# Align features
for col in feature_names:
    if col not in data:
        data[col] = 0

X = data[feature_names]
X_scaled = scaler.transform(X)

# Predict
prob = model.predict_proba(X_scaled)[0][1]
print(f"Prediction Probability: {prob:.2f}")