import pandas as pd
import pickle
import streamlit as st

# ===== LOAD MODELS =====
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("lgb_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

with open("cat_model.pkl", "rb") as f:
    cat_model = pickle.load(f)

with open("top_features.pkl", "rb") as f:
    feature_data = pickle.load(f)
    if isinstance(feature_data, dict):
        top_10_features = feature_data.get("features", feature_data)
    else:
        top_10_features = feature_data

# ===== MANUAL FRIENDLY PLACEHOLDER NAMES =====
# These will be shown in Streamlit regardless of raw column names
PLACEHOLDERS = [
    "Wood Density",
    "Moisture Content (%)",
    "Machine Speed (m/min)",
    "Blade Angle (°)",
    "Temperature (°C)",
    "Pressure (bar)",
    "Grain Direction",
    "Operator Experience (yrs)",
    "Log Diameter (cm)",
    "Cutting Depth (mm)"
]

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Top 10 Feature Ensemble Predictor", layout="wide")
st.title("Wood Prediction")

input_data = {}
for i, feature in enumerate(top_10_features):
    placeholder_label = PLACEHOLDERS[i] if i < len(PLACEHOLDERS) else feature
    input_data[feature] = st.number_input(placeholder_label, value=0.0)

if st.button("Predict"):
    new_df = pd.DataFrame([input_data])
    pred_xgb = xgb_model.predict(new_df)[0]
    pred_lgb = lgb_model.predict(new_df)[0]
    pred_cat = cat_model.predict(new_df)[0]
    final_pred = (0.4 * pred_xgb) + (0.3 * pred_lgb) + (0.3 * pred_cat)
    st.success(f"Predicted KD_BFOUT: {final_pred:.2f}")
