import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

numeric_columns = ['temperature', 'pressure', 'humidity', 'year']
#'day'
#Waarden die we nodig hebben voor het maken van sequences om het model te trainen
output_feature = "temperature"
output_feature_index = numeric_columns.index(output_feature)
#We willen 24 uur in de toekomst de temperatuur voorspellen
future_days = 24
#Hiervoor gebruiken we n_steps aantal uur
n_steps = 72
#Om te voorspellen gebruiken we de waarden uit numeric_columns, we moeten de length hebben van de array zodat het model weet hoeveel inputs het kan verwachten
n_inputs = len(numeric_columns)

def combine_data():
    folder_path = "./data/uurmetingen"
    output = "./data/hourly_data.txt"

    text_files = [file for file in os.listdir(folder_path)]

    with open(output, "w") as outfile:
        for txt_file in text_files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, "r") as infile:
                outfile.write(infile.read())

def create_pandas_frame() -> pd.DataFrame:
    data = "./data/hourly_data.txt"
    extracted_data = []
    with open(data, "r") as lines:
        for line in lines:
            values = line.split(",")
            date = datetime.strptime(values[1], "%Y%m%d")
            if int(values[2]) != 24:
                date = date.replace(hour=int(values[2]))
            else:
                date = date.replace(hour=0)
            temperature = int(values[7]) / 10
            pressure = int(values[14]) / 10
            humidity = int(values[17])

            data = {
                "date": date,
                "year": date.year,
                #"day": date.day,
                "hour_sin": np.sin(2 * np.pi * date.hour / 24),
                "hour_cos": np.cos(2 * np.pi * date.hour / 24),
                "month_sin": np.sin(2 * np.pi * date.month / 12),
                "month_cos": np.cos(2 * np.pi * date.month / 12),
                "pressure": pressure,
                'humidity': humidity,
                'temperature': temperature

            }
            extracted_data.append(data)
    
    hourly_dataframe = pd.DataFrame(extracted_data)

    #Zeker weten dat we alle numerieke dingen ook daadwerkelijk opslaan als nummer. Als iets niet als nummer opgeslagen kan worden, bijvoorbeeld None, wordt dit NaN.
    hourly_dataframe[numeric_columns] = hourly_dataframe[numeric_columns].apply(pd.to_numeric, errors='coerce')

    #Als index van ons DataFrame gebruiken we de datum
    hourly_dataframe.set_index('date', inplace=True)

    #Zeker maken dat we daadwerkelijk alles op datum gesorteerd hebben
    hourly_dataframe.sort_values(by='date', inplace=True)
    
    return hourly_dataframe

def prepare_data_for_training(data: pd.DataFrame):
    """
    Deze functie splitst alle data op in training, validatie en testing en normaliseert de waarden voor model training
    """
    split = 0.8
    training_size = int(len(data) * split)
    remaining_size = len(data) - training_size
    validation_size = remaining_size // 2
    testing_size = remaining_size - validation_size

    training_data = data[:training_size]
    validation_data = data[training_size:training_size + validation_size]
    testing_data = data[-testing_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))

    training_data_normalized = scaler.fit_transform(training_data[numeric_columns])
    validation_data_normalized = scaler.transform(validation_data[numeric_columns])
    testing_data_normalized = scaler.transform(testing_data[numeric_columns])

    return scaler, training_data_normalized, validation_data_normalized, testing_data_normalized


def save_dataframe(data: pd.DataFrame, path: str):
    data.to_csv(path, index=True)

def create_sequences(data_input, n_steps, future_days, out_feature_index):
    """
    Functie die sequences maakt van onze data die een LSTM kan gebruiken
    """
    x, y = [], []
    for i in range(len(data_input) - n_steps - future_days):
        end_ix = i + n_steps
        out_end_ix = end_ix + future_days
        if out_end_ix > len(data_input):
            break
        seq_x, seq_y = data_input[i:end_ix, :], data_input[out_end_ix - 1, out_feature_index]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)

def test_model(model, x_testing, y_testing, scaler):
    predictions = model.predict(x_testing)

    padded_predictions = np.zeros((predictions.shape[0], n_inputs))
    padded_predictions[:, output_feature_index] = predictions.flatten()
    print(padded_predictions)
    predictions_rescaled = scaler.inverse_transform(padded_predictions)[:, output_feature_index]
    print(predictions_rescaled)

    padded_y_testing = np.zeros((y_testing.shape[0], n_inputs))
    padded_y_testing[:, output_feature_index] = y_testing.flatten()
    y_testing_rescaled = scaler.inverse_transform(padded_y_testing)[:, output_feature_index]

    errors = predictions_rescaled - y_testing_rescaled
    mae = np.mean(np.abs(errors))
    print(f"Lijst van daadwerkelijke waarde vs voorspelde waarde: {errors}\nMean Absolute Error: {mae}")

def main():
    data_path = './data/weather_data_api_daily.csv'
    combine_data()
    dataframe = create_pandas_frame()
    save_dataframe(dataframe, data_path)

    scaler, training, validation, testing = prepare_data_for_training(dataframe)

    x_train, y_train = create_sequences(training, n_steps, future_days, output_feature_index)
    x_validation, y_validation = create_sequences(validation, n_steps, future_days, output_feature_index)
    x_testing, y_testing = create_sequences(testing, n_steps, future_days, output_feature_index)

    #Hier maken we het model dat we gaan trainen
    model = Sequential([
        LSTM(128,
             input_shape=(n_steps, x_train.shape[2])),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    #Callback die we gebruiken wanneer het model te inaccuraat wordt, hiermee proberen we overfitting te voorkomen
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_validation, y_validation),
        epochs=50,
        batch_size=128,
        callbacks=[early_stop]
    )

    model.save("./model/weather_model.h5")
    dump(scaler, './model/scaler.pkl')

    test_model(model, x_testing, y_testing, scaler)

if __name__ == '__main__': 
    main()