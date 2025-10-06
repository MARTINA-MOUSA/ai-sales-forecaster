import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Paths ---
MODEL_PATH = "models/xgb_sales_forecast.pkl"
SCALER_PATH = "models/scaler.pkl"
TEST_PATH = "data/test_sets/test_data.csv"

# --- Load Model, Scaler, Test Data ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
test_data = pd.read_csv(TEST_PATH)
print(f"Test data loaded: {test_data.shape}")

# --- Features & Target ---
features = [
    'Store','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment',
    'Year','Month','Week','IsWeekend','Season','IsYearEnd','Day','Quarter',
    'IsHolidayWeek','Temp_Fuel','CPI_Unemp','Month_sin','Month_cos',
    'Rolling_4w','Sales_diff_rolling','Cluster'
]
target = "Actual_Sales"

X_test = test_data[features]
y_test = test_data[target]

# --- Predict ---
y_pred = model.predict(X_test)

# --- Metrics ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation (Test Data):\nMAE={mae:.2f}\nRMSE={rmse:.2f}\nR²={r2:.3f}")

# --- Residual Analysis ---
residuals = y_test - y_pred
print(f"Mean residual: {residuals.mean():.2f}, Std residual: {residuals.std():.2f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True, color="skyblue")
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
