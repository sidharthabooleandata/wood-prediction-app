import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ===== CONFIG =====
FILE_PATH = r"C:\Users\sidhartha-BD\Desktop\wood\1Y02_18Mon_Model.xlsx"
TARGET_COL = "KD_BFOUT"

# ===== FRIENDLY LABELS MASTER MAP =====
FRIENDLY_LABELS = {
    "KD_DENSITY": "Wood Density",
    "KD_MOISTURE": "Moisture Content (%)",
    "KD_SPEED": "Machine Speed (m/min)",
    "KD_BLADE": "Blade Angle (°)",
    "KD_TEMPERATURE": "Temperature (°C)",
    "KD_PRESSURE": "Pressure (bar)",
    "KD_GRAIN": "Grain Direction",
    "KD_OPERATOR": "Operator Experience (yrs)",
    "KD_DIAMETER": "Log Diameter (cm)",
    "KD_DEPTH": "Cutting Depth (mm)"
}

# ===== LOAD DATA =====
def load_data():
    df = pd.read_excel(FILE_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' not found. Available: {df.columns.tolist()}")
    df = df.dropna(subset=[TARGET_COL]).copy()

    # Convert datetime columns
    date_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9

    # Encode categoricals
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()
X_full = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ===== GET TOP 10 FEATURES =====
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
base_model = XGBRegressor(
    random_state=42, n_estimators=500, learning_rate=0.1, max_depth=8, colsample_bytree=0.8
)
base_model.fit(X_train_full, y_train_full)

feat_imp = pd.Series(base_model.feature_importances_, index=X_full.columns).sort_values(ascending=False)
top_10_features = feat_imp.head(10).index.tolist()

# ===== TRAIN ENSEMBLE MODELS =====
X_top = X_full[top_10_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_top, y, test_size=0.2, random_state=42
)

xgb_model = XGBRegressor(
    n_estimators=1500, learning_rate=0.05, max_depth=8, colsample_bytree=0.9,
    subsample=0.9, random_state=42, n_jobs=-1, tree_method="hist"
)
xgb_model.fit(X_train, y_train)

lgb_model = LGBMRegressor(
    n_estimators=1500, learning_rate=0.05, max_depth=-1, num_leaves=31,
    colsample_bytree=0.9, subsample=0.9, random_state=42, n_jobs=-1
)
lgb_model.fit(X_train, y_train)

cat_model = CatBoostRegressor(
    iterations=1500, learning_rate=0.05, depth=8, random_state=42, verbose=0
)
cat_model.fit(X_train, y_train)

# ===== ENSEMBLE PREDICTIONS =====
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_cat = cat_model.predict(X_test)
y_pred_ensemble = (0.4 * y_pred_xgb) + (0.3 * y_pred_lgb) + (0.3 * y_pred_cat)

# ===== METRICS =====
try:
    rmse = mean_squared_error(y_test, y_pred_ensemble, squared=False)
except TypeError:
    rmse = mean_squared_error(y_test, y_pred_ensemble) ** 0.5
mae = mean_absolute_error(y_test, y_pred_ensemble)
r2 = r2_score(y_test, y_pred_ensemble)
accuracy = r2 * 100

print("\n===== MODEL PERFORMANCE (Ensemble) =====")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print(f"Accuracy: {accuracy:.2f}%")
print("========================================\n")

# ===== SAVE MODELS & FEATURE LABELS =====
friendly_map = {feat: FRIENDLY_LABELS.get(feat, feat) for feat in top_10_features}

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("lgb_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)

with open("cat_model.pkl", "wb") as f:
    pickle.dump(cat_model, f)

with open("top_features.pkl", "wb") as f:
    pickle.dump({"features": top_10_features, "labels": friendly_map}, f)

print("✅ Models and friendly feature labels saved successfully!")
