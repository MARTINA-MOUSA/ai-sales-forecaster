
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from monitoring import log_system_metrics
import threading

os.makedirs("logs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)
logging.info('Started data_preprocessing.py')

# === تشغيل مراقبة النظام في الخلفية (اختياري) ===
monitor_thread = threading.Thread(target=log_system_metrics, kwargs={'interval': 5, 'duration': 60})
monitor_thread.start()

try:
    data = pd.read_csv("data/raw/Walmart.csv")
    data.columns = data.columns.str.strip()
    logging.info(f" Loaded Walmart.csv successfully with shape {data.shape}")
except Exception as e:
    logging.error(f" Error loading Walmart.csv: {e}")
    raise

# === التحقق من وجود عمود التاريخ ===
if 'Date' not in data.columns:
    logging.error(f" Column 'Date' not found! Columns: {data.columns.tolist()}")
    raise KeyError(f"Column 'Date' not found!")

# === معالجة عمود التاريخ ===
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
missing_dates = data['Date'].isna().sum()
if missing_dates > 0:
    logging.warning(f" {missing_dates} invalid dates found — dropped.")
    data = data.dropna(subset=['Date'])
logging.info(" Date column processed successfully.")

# === مراقبة القيم المفقودة ===
missing_counts = data.isna().sum()
logging.info(f" Missing values per column:\n{missing_counts}")

# === إنشاء ميزات زمنية ===
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['DayOfMonth'] = data['Date'].dt.day
data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
data['Season'] = (data['Month'] % 12 + 3) // 3
data['IsYearEnd'] = data['Month'].isin([11, 12]).astype(int)

# === إزالة القيم الشاذة ===
Q1 = data['Weekly_Sales'].quantile(0.25)
Q3 = data['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
initial_len = len(data)
data = data[(data['Weekly_Sales'] >= lower) & (data['Weekly_Sales'] <= upper)].reset_index(drop=True)
logging.info(f"🧹 Outliers removed: {initial_len - len(data)} rows")

# === إنشاء ميزات إضافية ===
data['Temp_Fuel'] = data['Temperature'] * data['Fuel_Price']
data['CPI_Unemp'] = data['CPI'] * data['Unemployment']
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# === ميزات متحركة (Rolling Features) ===
data['Rolling_4w'] = data.groupby('Store')['Weekly_Sales'].transform(lambda x: x.rolling(4, min_periods=1).mean())
data['Sales_diff_rolling'] = data['Weekly_Sales'] - data['Rolling_4w']

# === تجميع المتاجر وعمل Clustering ===
store_features = data.groupby('Store').agg({
    'Weekly_Sales': ['mean', 'std', 'max'],
    'Holiday_Flag': 'mean',
    'Temperature': 'mean',
    'Fuel_Price': 'mean'
}).round(2)

store_features.columns = ['_'.join(col).strip() for col in store_features.columns.values]
scaler = StandardScaler()
store_scaled = scaler.fit_transform(store_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
store_features['Cluster'] = kmeans.fit_predict(store_scaled)
data = data.merge(store_features[['Cluster']], left_on='Store', right_index=True)

# === ترميز الأعمدة ===
le = LabelEncoder()
data['Store'] = le.fit_transform(data['Store'])

# === اختيار الخصائص والهدف ===
features = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
    'Unemployment', 'Year', 'Month', 'Week', 'IsWeekend', 'Season', 'IsYearEnd',
    'Temp_Fuel', 'CPI_Unemp', 'Month_sin', 'Month_cos', 'Rolling_4w', 'Sales_diff_rolling', 'Cluster'
]
target = 'Weekly_Sales'
X = data[features]
y = data[target]

logging.info(f" Processed data shape: {data.shape}")
logging.info(f" Features used: {features}")

# === حفظ البيانات المعالجة ===
output_path = "data/processed/walmart_enhanced.csv"
data.to_csv(output_path, index=False)
logging.info(f" Saved processed dataset to {output_path}")

sns.histplot(data['Weekly_Sales'], bins=30, kde=True)
plt.title("Sales Distribution (After Enhanced Features)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data[features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

monitor_thread.join()
logging.info(" Data preprocessing and monitoring completed successfully.")
print("Data preprocessing completed successfully!")
