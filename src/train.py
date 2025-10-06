import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import logging
from logging.handlers import RotatingFileHandler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from monitoring import log_system_metrics
import mlflow
import mlflow.sklearn
from datetime import datetime

# === Setup Logging ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)
handler = RotatingFileHandler("logs/train.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
logging.getLogger().addHandler(handler)

logging.info("Started train.py")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AI_Sales_Forecaster")

# === Load Data ===
DATA_PATH = "data/processed/walmart_enhanced.csv"
try:
    data = pd.read_csv(DATA_PATH)
    logging.info(f"Data loaded successfully: {data.shape}")
except Exception as e:
    logging.error(f"Error loading {DATA_PATH}: {e}")
    raise

# === Features & Target ===
features = [
    'Store','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment',
    'Year','Month','Week','IsWeekend','Season','IsYearEnd',
    'IsHolidayWeek','Temp_Fuel','CPI_Unemp','Month_sin','Month_cos',
    'Rolling_4w','Sales_diff_rolling','Cluster'
]
target = 'Weekly_Sales'

existing_features = [col for col in features if col in data.columns]
if len(existing_features) < len(features):
    missing = set(features) - set(existing_features)
    logging.warning(f"Some features are missing: {missing}")

# === Remove non-numeric columns before scaling ===
X = data[existing_features].select_dtypes(include=[np.number])
y = data[target]

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
logging.info(f"Train/Test split: {X_train.shape}/{X_test.shape}")

# === Scale Data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Save Scaler ===
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
logging.info("Scaler saved -> models/scaler.pkl")

# === Save Train/Test Sets ===
os.makedirs("data/train_sets", exist_ok=True)
os.makedirs("data/test_sets", exist_ok=True)

train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_df["Actual_Sales"] = y_train.values
train_df.to_csv("data/train_sets/train_data.csv", index=False)

test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_df["Actual_Sales"] = y_test.values
test_df.to_csv("data/test_sets/test_data.csv", index=False)

logging.info("Train/Test CSVs saved -> data/train_sets/train_data.csv & data/test_sets/test_data.csv")

# === Train Model ===
params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}
model = XGBRegressor(**params)

with mlflow.start_run(run_name=f"train_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.log_params(params)

    logging.info("Starting model training...")
    system_metrics_before = log_system_metrics()
    if system_metrics_before:
        mlflow.log_metrics({
            "CPU_Before": system_metrics_before.get("cpu_usage", 0),
            "RAM_Before": system_metrics_before.get("memory_usage", 0)
        })

    model.fit(X_train_scaled, y_train)
    logging.info("Model training completed successfully.")

    system_metrics_after = log_system_metrics()
    if system_metrics_after:
        mlflow.log_metrics({
            "CPU_After": system_metrics_after.get("cpu_usage", 0),
            "RAM_After": system_metrics_after.get("memory_usage", 0)
        })

    # === Evaluate Model ===
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    mlflow.log_metrics(metrics)
    logging.info(f"Model evaluation: {metrics}")

    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # === Save Model & Log to MLflow ===
    joblib.dump(model, "models/xgb_sales_forecast.pkl")
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_test.iloc[:2]
    )
    mlflow.log_artifact("models/metrics.json")
    mlflow.log_artifact("models/scaler.pkl")

    logging.info("Model and artifacts logged to MLflow.")

print(f"\n Training finished successfully! MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
