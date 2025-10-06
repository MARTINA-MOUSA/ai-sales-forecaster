import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from predict import make_prediction  # use your existing prediction logic

# --- Load model and scaler ---
model_path = os.path.join("models", "xgb_sales_forecast.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Model or scaler not found. Please train the model first.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title(" AI Sales Forecasting App")
st.markdown("Enter store and economic details to predict **Weekly Sales**")

# --- User input fields ---
store = st.number_input("Store ID", min_value=1, max_value=45, value=1)
holiday_flag = st.selectbox("Holiday Flag", [0, 1])
temperature = st.number_input("Temperature (°F)", value=60.0)
fuel_price = st.number_input("Fuel Price ($)", value=3.0)
cpi = st.number_input("CPI", value=200.0)
unemployment = st.number_input("Unemployment Rate", value=7.0)
year = st.number_input("Year", min_value=2010, max_value=2030, value=2012)
month = st.slider("Month", 1, 12, 6)
week = st.slider("Week", 1, 52, 26)
is_weekend = st.selectbox("Is Weekend?", [0, 1])
season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
is_year_end = st.selectbox("Is Year End?", [0, 1])
rolling_4w = st.number_input("Rolling 4 Weeks Avg Sales", value=200000.0)
sales_diff_rolling = st.number_input("Sales Diff Rolling", value=5000.0)
cluster = st.number_input("Cluster ID", min_value=0, max_value=10, value=1)

# --- Feature engineering (same as in train.py) ---
day = 15
quarter = (month - 1) // 3 + 1
is_holiday_week = 1 if holiday_flag == 1 else 0
temp_fuel = temperature * fuel_price
cpi_unemp = cpi * unemployment
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# --- Encode categorical features ---
season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
season_encoded = season_mapping.get(season, 0)

# --- Build input DataFrame ---
input_data = pd.DataFrame([{
    'Store': store,
    'Holiday_Flag': holiday_flag,
    'Temperature': temperature,
    'Fuel_Price': fuel_price,
    'CPI': cpi,
    'Unemployment': unemployment,
    'Year': year,
    'Month': month,
    'Week': week,
    'IsWeekend': is_weekend,
    'Season': season_encoded,
    'IsYearEnd': is_year_end,
    'Day': day,
    'Quarter': quarter,
    'IsHolidayWeek': is_holiday_week,
    'Temp_Fuel': temp_fuel,
    'CPI_Unemp': cpi_unemp,
    'Month_sin': month_sin,
    'Month_cos': month_cos,
    'Rolling_4w': rolling_4w,
    'Sales_diff_rolling': sales_diff_rolling,
    'Cluster': cluster
}])

# --- Predict ---
if st.button("🔮 Predict Weekly Sales"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Weekly Sales: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
