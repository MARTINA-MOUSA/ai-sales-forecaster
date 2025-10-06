import pandas as pd
import joblib
import os

# تحديد مسارات الملفات
model_path = os.path.join("models", "xgb_sales_forecast.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

# التحقق من وجود الملفات
if not os.path.exists(model_path):
    raise FileNotFoundError(" Model file not found. Please train the model first.")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(" Scaler file not found. Please train the model first.")

# تحميل النموذج والمقياس (scaler)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def make_prediction(input_data):
    """
    input_data: dict أو DataFrame يحتوي على نفس الأعمدة اللي استخدمتها أثناء التدريب
    return: DataFrame يحتوي على التوقعات
    """

    # تحويل البيانات إلى DataFrame لو كانت dict
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = pd.DataFrame(input_data)

    # تطبيق الـ scaler لو استخدمته في التدريب
    input_scaled = scaler.transform(input_df)

    # عمل التنبؤ
    predictions = model.predict(input_scaled)

    # إرجاع النتائج مع البيانات الأصلية
    input_df["Predicted_Sales"] = predictions
    return input_df
