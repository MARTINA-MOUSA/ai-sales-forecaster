import joblib
import os
import logging
import pandas as pd

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

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)
logging.info('Started predict.py')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info('Model and scaler loaded successfully.')
except Exception as e:
    logging.error(f'Error loading model or scaler: {e}')
    raise

def make_prediction(input_data):
    try:
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
        input_scaled = scaler.transform(input_df)
        predictions = model.predict(input_scaled)
        input_df["Predicted_Sales"] = predictions
        logging.info('Prediction made successfully.')
        return input_df
    except Exception as e:
        logging.error(f'Error in make_prediction: {e}')
        raise
