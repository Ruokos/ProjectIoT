from keras.models import load_model, Sequential
from keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import sqlite3 as sql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

numeric_columns = ['temperature', 'pressure', 'humidity', 'year', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

#How many hours the model expects to predict the next value
LIMIT_ROWS = 168

weather_model: Sequential = load_model('./model/weather_model.h5')
scalers = load('./model/scalers.pkl')

conn = sql.connect('./db/sensor_readings')

#Select last 168 rows (last 168 hours of sensor readings) from database
query = f"""
SELECT *
FROM data
ORDER BY id DESC
LIMIT {LIMIT_ROWS};
"""

#Data uit database laden in dataframe
df = pd.read_sql_query(query, conn)

#Data in hetzelfde format zetten als waarmee het model is getrained
df['date'] = pd.to_datetime(df['date'])
df['hour_sin'] = np.sin(2 * np.pi * df['date'].hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

#Als index van ons DataFrame gebruiken we de datum
df.set_index('date', inplace=True)
df.sort_values(by='date', inplace=True)

#Scalers die gebruikt waren tijdens trainen toepassen op sensordata
model_input = df[numeric_columns].copy()
for column in numeric_columns:
    model_input[column] = scalers[column].transform(df[[column]])

#Model temperatuur laten voorspellen
prediction = weather_model.predict(model_input.to_numpy())
prediction_rescale = scalers['temperature'].inverse_transform(prediction.reshape(-1, -1))

latest_datetime = df.index[-1]
prediction_time = latest_datetime + timedelta(hours=24)
print(f"Predicted temperature at {prediction_time}: {prediction_rescale[0][0]:.1f} °C")