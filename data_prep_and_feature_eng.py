import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
import mlflow

# Turn off oneDNN messages the below line should be written before tesnsorflow imports:
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping



# ignoring warnings
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv('./Dataset/train.csv', parse_dates=['date'])
holidays = pd.read_csv('./Dataset/holidays_events.csv', parse_dates=['date'])
oil = pd.read_csv('./Dataset/oil.csv', parse_dates=['date'])
stores = pd.read_csv('./Dataset/stores.csv')
transactions = pd.read_csv('./Dataset/transactions.csv', parse_dates=['date'])

# Merge datasets, ensuring 'type' column is preserved
df = train.merge(stores, on='store_nbr', how='left')
df = df.merge(holidays, on='date', how='left')
df = df.merge(oil, on='date', how='left')
# Include 'store_nbr' in the merge with transactions to keep 'type'
df = df.merge(transactions, on=['date', 'store_nbr'], how='left')


# Convert categorical features
# Access it using df['type_x'] if there's a naming conflict after the merge
df['holiday_type'] = df['type_x'].apply(lambda x: 1 if x != 'None' else 0)
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day

# Aggregate sales by date
daily_sales = df.groupby('date').agg({
    'sales': 'sum',
    'transactions': 'sum',
    'dcoilwtico': 'mean',
    'holiday_type': 'max',
    'is_weekend': 'max',
    'month': 'max',
    'day_of_week': 'max',
    'day_of_month': 'max'
}).reset_index()

# Create lag features
for lag in [1, 7, 14, 30]:
    daily_sales[f'sales_lag_{lag}'] = daily_sales['sales'].shift(lag)

# Drop missing values from lag features
daily_sales = daily_sales.dropna()

# Split into train and test (last 30 days for testing)
train = daily_sales.iloc[:-30]
test = daily_sales.iloc[-30:]

X_train = train.drop(columns=['sales', 'date'])
y_train = train['sales']
X_test = test.drop(columns=['sales', 'date'])
y_test = test['sales']