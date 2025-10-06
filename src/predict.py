import pandas as pd
import joblib
import os

model_path = os.path.join("models", "xgb_sales_forecast.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(" Model file not found. Please train the model first.")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(" Scaler file not found. Please train the model first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def make_prediction(input_data):

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    predictions = model.predict(input_scaled)

    input_df["Predicted_Sales"] = predictions
    return input_df
