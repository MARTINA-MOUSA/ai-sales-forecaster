from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os
from src.predict import make_prediction  # import your prediction function

app = FastAPI(
    title="Sales Forecasting API",
    description=" FastAPI service for predicting sales using the trained XGBoost model",
    version="1.0"
)

# Load model and scaler
model_path = os.path.join("models", "xgb_sales_forecast.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(" Model file not found.")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(" Scaler file not found.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


@app.get("/")
def root():
    return {"message": "Welcome to the AI Sales Forecasting API "}


@app.post("/predict_single/")
def predict_single(data: dict):
    """
    Predict sales for a single input record (JSON format)
    """
    result = make_prediction(data)
    return result.to_dict(orient="records")


@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file to get predictions for all rows
    """
    df = pd.read_csv(file.file)
    result = make_prediction(df)
    output_path = os.path.join("data", "processed", "predictions.csv")
    result.to_csv(output_path, index=False)
    return {
        "message": "  Predictions generated successfully!",
        "saved_to": output_path,
        "rows": len(result)
    }
