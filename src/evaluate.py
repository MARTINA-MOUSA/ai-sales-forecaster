import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from monitoring import log_system_metrics  # نستدعي ملف المراقبة
import threading

os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# === Logging Setup ===
logging.basicConfig(
    filename='logs/evaluate.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)
logging.info('Started evaluate.py')

# --- Paths ---
MODEL_PATH = "models/xgb_sales_forecast.pkl"
SCALER_PATH = "models/scaler.pkl"
TEST_PATH = "data/test_sets/test_data.csv"

# --- Load Model, Scaler, and Test Data ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    test_data = pd.read_csv(TEST_PATH)
    logging.info(f"Test data loaded successfully with shape: {test_data.shape}")
except Exception as e:
    logging.error(f"Error loading model/scaler/test data: {e}")
    raise

# --- Features & Target ---
features = [
    'Store','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment',
    'Year','Month','Week','IsWeekend','Season','IsYearEnd','Day','Quarter',
    'IsHolidayWeek','Temp_Fuel','CPI_Unemp','Month_sin','Month_cos',
    'Rolling_4w','Sales_diff_rolling','Cluster'
]
target = "Actual_Sales"

existing_features = [col for col in features if col in test_data.columns]
if len(existing_features) < len(features):
    missing = set(features) - set(existing_features)
    logging.warning(f"Missing features from test data: {missing}")
    print("Warning: Missing features:", missing)

X_test = test_data[existing_features]
y_test = test_data[target]

# --- Start System Monitoring (in background) ---
monitor_thread = threading.Thread(target=log_system_metrics, kwargs={'interval': 5, 'duration': 60})
monitor_thread.start()

# --- Predict ---
try:
    y_pred = model.predict(X_test)
    logging.info("Prediction completed successfully.")
except Exception as e:
    logging.error(f"Error during prediction: {e}")
    raise

# --- Metrics ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

logging.info(f"Model Evaluation → MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
print(f"\nModel Evaluation (Test Data):\nMAE={mae:.2f}\nRMSE={rmse:.2f}\nR²={r2:.3f}")

# --- Residual Analysis ---
residuals = y_test - y_pred
logging.info(f"Residuals: mean={residuals.mean():.2f}, std={residuals.std():.2f}")
print(f"Mean residual: {residuals.mean():.2f}, Std residual: {residuals.std():.2f}")

# --- Save Results ---
results_path = f"reports/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residual": residuals
}).to_csv(results_path, index=False)
logging.info(f"Saved evaluation results to {results_path}")

# --- Visualization ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("reports/actual_vs_predicted.png")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True, color="skyblue")
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/residual_distribution.png")
plt.show()

monitor_thread.join()
logging.info("Evaluation and monitoring completed successfully.")
