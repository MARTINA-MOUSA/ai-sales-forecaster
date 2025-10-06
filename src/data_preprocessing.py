import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Raw Data ===
data = pd.read_csv("data/raw/Walmart.csv")
data.columns = data.columns.str.strip()  # إزالة أي مسافات في أسماء الأعمدة

if 'Date' not in data.columns:
    raise KeyError(f" Column 'Date' not found! Columns available: {data.columns.tolist()}")

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# لو في تواريخ مش اتقرت
missing_dates = data['Date'].isna().sum()
if missing_dates > 0:
    print(f" Warning: {missing_dates} dates could not be parsed. They will be dropped.")
    data = data.dropna(subset=['Date'])

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['DayOfMonth'] = data['Date'].dt.day
data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
data['Season'] = (data['Month'] % 12 + 3) // 3
data['IsYearEnd'] = data['Month'].isin([11, 12]).astype(int)

Q1 = data['Weekly_Sales'].quantile(0.25)
Q3 = data['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
data = data[(data['Weekly_Sales'] >= lower) & (data['Weekly_Sales'] <= upper)].reset_index(drop=True)

data['Temp_Fuel'] = data['Temperature'] * data['Fuel_Price']
data['CPI_Unemp'] = data['CPI'] * data['Unemployment']

# Cyclical features for Month
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# Rolling features: mean of last 4 weeks for each store
data['Rolling_4w'] = data.groupby('Store')['Weekly_Sales'].transform(lambda x: x.rolling(4, min_periods=1).mean())
data['Sales_diff_rolling'] = data['Weekly_Sales'] - data['Rolling_4w']

# === Store Clustering ===
store_features = data.groupby('Store').agg({
    'Weekly_Sales': ['mean', 'std', 'max'],
    'Holiday_Flag': 'mean',
    'Temperature': 'mean',
    'Fuel_Price': 'mean'
}).round(2)

# Flatten MultiIndex columns
store_features.columns = ['_'.join(col).strip() for col in store_features.columns.values]

scaler = StandardScaler()
store_scaled = scaler.fit_transform(store_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
store_features['Cluster'] = kmeans.fit_predict(store_scaled)

# Merge cluster info back to main dataset
data = data.merge(store_features[['Cluster']], left_on='Store', right_index=True)

# === Encode Store as categorical number for model ===
le = LabelEncoder()
data['Store'] = le.fit_transform(data['Store'])

# === Select Features and Target ===
features = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
    'Unemployment', 'Year', 'Month', 'Week', 'IsWeekend', 'Season', 'IsYearEnd',
    'Temp_Fuel', 'CPI_Unemp', 'Month_sin', 'Month_cos', 'Rolling_4w', 'Sales_diff_rolling', 'Cluster'
]
target = 'Weekly_Sales'

X = data[features]
y = data[target]

# === Optional: save processed dataset ===
data.to_csv("data/processed/walmart_enhanced.csv", index=False)
print(" Processed dataset saved to data/processed/walmart_enhanced.csv")
print("Features for model:", X.columns.tolist())
print("Target:", target)

# === Quick Check Plots ===
plt.figure(figsize=(8,5))
sns.histplot(data['Weekly_Sales'], bins=30, kde=True)
plt.title("Sales Distribution (After Enhanced Features)")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data[features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
