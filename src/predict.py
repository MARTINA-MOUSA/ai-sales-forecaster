import joblib
import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import time
from monitoring import log_system_metrics  

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)
handler = RotatingFileHandler('logs/predict.log', maxBytes=5000000, backupCount=5)
logging.getLogger().addHandler(handler)

# تحميل الموديل والسكيلر
model_path = os.path.join("models", "xgb_sales_forecast.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info('Model and scaler loaded successfully.')
except Exception as e:
    logging.error(f'Error loading model or scaler: {e}')
    raise

def make_prediction(input_data):
    try:
        logging.info(f'Received input: {input_data}')

        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)

        start_time = time.time()
        input_scaled = scaler.transform(input_df)
        predictions = model.predict(input_scaled)
        elapsed_time = time.time() - start_time

        input_df["Predicted_Sales"] = predictions

        # سجل المتركس بتاعة الجهاز
        system_metrics = get_system_metrics()
        logging.info(f'System Metrics: {system_metrics}')

        # سجل تفاصيل التنبؤ
        logging.info(f'Prediction done in {elapsed_time:.2f}s. Output: {predictions.tolist()}')
        return input_df
    except Exception as e:
        logging.error(f'Error in make_prediction: {e}')
        raise
