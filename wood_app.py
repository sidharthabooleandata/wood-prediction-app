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

# ===== LOAD TOP FEATURES =====
with open("top_features.pkl", "rb") as f:
    feature_data = pickle.load(f)
    if isinstance(feature_data, dict):
        top_10_features = feature_data.get("features", feature_data)
    else:
        top_10_features = feature_data

# ===== FRIENDLY LABELS =====
FRIENDLY_LABELS = {
    "Prod_order": "Production Order",
    "Plant": "Plant",
    "Order_Type": "Order Type",

    # Input side (KS)
    "KS_Material": "Input Material",
    "KS_MaterialSpecie": "Input Species",
    "KS_MaterialThickness": "Input Thickness (mm)",
    "KS_mvt_type": "Input Movement Type",
    "KS_Postingdt": "Input Posting Date",
    "KS_ProcessState": "Input Process State",
    "KS_TGrade": "Input Timber Grade",
    "KS_TLength": "Input Timber Length (mm)",
    "KS_TWidth": "Input Timber Width (mm)",
    "KS_BFIN": "Input Volume (Board Feet)",

    # Output side (KD)
    "KD_Material": "Output Material",
    "KD_MaterialSpecie": "Output Species",
    "KD_MaterialThickness": "Output Thickness (mm)",
    "KD_mvt_type": "Output Movement Type",
    "KD_Postingdt": "Output Posting Date",
    "KD_ProcessState": "Output Process State",
    "KD_TGrade": "Output Timber Grade",
    "KD_TLength": "Output Timber Length (mm)",
    "KD_TWidth": "Output Timber Width (mm)",
    "KD_BFOUT": "Output Volume (Board Feet)"   # 🎯 Target
}

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Wood Prediction App", layout="wide")
st.title("Predict Output Volume (KD_BFOUT)")

input_data = {}
for feature in top_10_features:
    label = FRIENDLY_LABELS.get(feature, feature)   # fallback to raw name
    input_data[feature] = st.number_input(label, value=0.0)

if st.button("Predict"):
    new_df = pd.DataFrame([input_data])
    pred_xgb = xgb_model.predict(new_df)[0]
    pred_lgb = lgb_model.predict(new_df)[0]
    pred_cat = cat_model.predict(new_df)[0]
    final_pred = (0.4 * pred_xgb) + (0.3 * pred_lgb) + (0.3 * pred_cat)
    st.success(f"Predicted Output Volume (Board Feet): {final_pred:.2f}")
