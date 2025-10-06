# === train.py ===
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# --- Load Data ---
DATA_PATH = "data/processed/walmart_enhanced.csv"
data = pd.read_csv(DATA_PATH)
print(f"Data loaded: {data.shape}")

# --- Feature Engineering (ensure all enhanced features exist) ---
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Quarter'] = data['Date'].dt.quarter
data['IsHolidayWeek'] = data['Holiday_Flag'].apply(lambda x: 1 if x==1 else 0)
data['Temp_Fuel'] = data['Temperature'] * data['Fuel_Price']
data['CPI_Unemp'] = data['CPI'] * data['Unemployment']

# --- Select Features & Target ---
features = [
    'Store','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment',
    'Year','Month','Week','IsWeekend','Season','IsYearEnd','Day','Quarter',
    'IsHolidayWeek','Temp_Fuel','CPI_Unemp','Month_sin','Month_cos',
    'Rolling_4w','Sales_diff_rolling','Cluster'
]
target = 'Weekly_Sales'

X = data[features]
y = data[target]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Train/Test split → {X_train.shape}/{X_test.shape}")

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Save Scaler ---
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved → models/scaler.pkl")

# --- Save Train/Test Data ---
train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_df["Actual_Sales"] = y_train.values
test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_df["Actual_Sales"] = y_test.values

os.makedirs("data/train_sets", exist_ok=True)
os.makedirs("data/test_sets", exist_ok=True)
train_df.to_csv("data/train_sets/train_data.csv", index=False)
test_df.to_csv("data/test_sets/test_data.csv", index=False)
print("Train/Test sets saved successfully")

# --- Train XGBoost Model ---
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("XGBoost training completed")

# --- Evaluate on Test ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTest Evaluation:\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

# --- Save Model ---
joblib.dump(model, "models/xgb_sales_forecast.pkl")
print("Model saved → models/xgb_sales_forecast.pkl")
