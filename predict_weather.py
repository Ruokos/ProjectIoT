from keras.models import load_model
from keras.losses import MeanSquaredError
from joblib import load
import sqlite3 as sql
import pandas as pd

#How many hours the model expects to predict the next value
LIMIT_ROWS = 168

weather_model = load_model('./model/weather_model.h5')
scaler = load('./model/scaler.pkl')

conn = sql.connect('./db/sensor_readings')

#Select last 168 rows (last 168 hours of sensor readings)
query = f"""
SELECT *
FROM data
ORDER BY id DESC
LIMIT {LIMIT_ROWS};
"""

df = pd.read_sql_query(query, conn)
df = df.iloc[::-1]
model_input = df.to_numpy()

prediction = weather_model.predict(model_input)
prediction_rescale = scaler.inverse_transform(prediction.reshape(-1, -1))

print(f"Predicted temperature over 24 hours: {prediction_rescale}")