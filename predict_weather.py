from keras.models import load_model
from keras.losses import MeanSquaredError
from joblib import load
import sqlite3 as sql
import pandas as pd
import numpy as np
from datetime import datetime

numeric_columns = ['temperature', 'pressure', 'humidity', 'year', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

#How many hours the model expects to predict the next value
LIMIT_ROWS = 168

weather_model = load_model('./model/weather_model.h5')
scalers = load('./model/scalers.pkl')

conn = sql.connect('./db/sensor_readings')

#Select last 168 rows (last 168 hours of sensor readings) from database
query = f"""
SELECT *
FROM data
ORDER BY id DESC
LIMIT {LIMIT_ROWS};
"""

#Load queried database rows as dataframe
df = pd.read_sql_query(query, conn)
df = df.iloc[::-1]

df['datetime'] = pd.to_datetime(df['datetime'])

df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)

model_input = df[numeric_columns].copy()
for column in numeric_columns:
    model_input[column] = scalers[column].transform(df[[column]])

prediction = weather_model.predict(model_input.to_numpy())
prediction_rescale = scalers['temperature'].inverse_transform(prediction.reshape(-1, -1))

print(f"Predicted temperature over 24 hours: {prediction_rescale}")