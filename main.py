import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

np.set_printoptions(threshold=75)

numeric_columns = ['temperature', 'pressure', 'humidity', 'year', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

#Dit is het veld uit numeric_columns die we uiteindelijk willen voorspellen
output_feature = "temperature"
#Dit is de index van de waarde in de kolom. In het geval van temperatuur, is dat index 0
output_feature_index = numeric_columns.index(output_feature)
#Op hoeveel decimalen we uiteindelijk afronden
output_decimals = 1
#We willen 24 uur in de toekomst de temperatuur voorspellen
future_hours = 24
#Hiervoor gebruiken we n_steps aantal uur
n_steps = 168
#Om te voorspellen gebruiken we de waarden uit numeric_columns, we moeten de length hebben van de array zodat het model weet hoeveel inputs het kan verwachten
n_inputs = len(numeric_columns)

def combine_data():
    folder_path = "./data/uurmetingen"
    output = "./data/hourly_data.txt"

    #Een list maken van alle bestanden in onze /data/uurmetingen folder
    text_files = [file for file in os.listdir(folder_path)]

    #Alle bestanden uit de text_files list combineren tot 1 tekstbestand in /data/hourly_data.txt
    with open(output, "w") as outfile:
        for txt_file in text_files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, "r") as infile:
                outfile.write(infile.read())

def create_pandas_frame() -> pd.DataFrame:
    data = "./data/hourly_data.txt"
    extracted_data = []
    #Tekstbestand hourly_data.txt openen
    with open(data, "r") as lines:
        #Alle regels bij langs gaan
        for line in lines:
            #Alle waarden van een regel op de comma splitsen, en de waarden die we willen opslaan in een dictionary stoppen
            values = line.split(",")
            date = datetime.strptime(values[1], "%Y%m%d")
            if int(values[2]) != 24:
                date = date.replace(hour=int(values[2]))
            else:
                date = date.replace(hour=0)
            temperature = float(values[7]) / 10
            pressure = float(values[14]) / 10
            humidity = float(values[17])

            #Dictionary maken van alle gegevens die we willen opslaan van een regel
            data = {
                'temperature': temperature,
                "pressure": pressure,
                'humidity': humidity,
                "year": date.year,
                "hour_sin": np.sin(2 * np.pi * date.hour / 24),
                "hour_cos": np.cos(2 * np.pi * date.hour / 24),
                "month_sin": np.sin(2 * np.pi * date.month / 12),
                "month_cos": np.cos(2 * np.pi * date.month / 12),
                "date": date,
                #"day": date.day,
            }
            #Gemaakte dictionary toevoegen aan onze eindlijst van dictionaries
            #extracted_data heeft uiteindelijk elke regel van hourly_data.txt als dictionary
            extracted_data.append(data)
    
    #Dataframe maken van onze dictionaries
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
    #Split is hoeveel % van de dataset voor training wordt gebruikt. Een split van 0.8 betekent 80% van de data voor trainen
    split = 0.8

    #Kijken waar we onze dataframe in stukken moeten knippen aan de hand van de waarde split
    training_size = int(len(data) * split)
    remaining_size = len(data) - training_size
    validation_size = remaining_size // 2
    testing_size = remaining_size - validation_size

    #Slicing gebruiken om ons dataframe in drie stukken te delen
    training_data = data[:training_size]
    validation_data = data[training_size:training_size + validation_size]
    testing_data = data[-testing_size:]

    training_data_normalized = training_data.copy()
    validation_data_normalized = validation_data.copy()
    testing_data_normalized = testing_data.copy()

    #Scaler initisialisen als None, in de volgende loop slaan we hierin de scaler van temperatuur op zodat we deze later in de code kunnen hergebruiken
    scaler = None

    #Voor elke kolom in onze dataset, scalen we de waarden met een scaler tussen 0 en 1 voor het trainen van het model
    for column in numeric_columns:
        #Temporary scaler, gebruiken we in deze loop
        temp_scaler = MinMaxScaler(feature_range=(0, 1))

        #Gemaakte temp_scaler toepassen op onze datasets
        training_data_normalized[column] = temp_scaler.fit_transform(training_data[[column]])
        validation_data_normalized[column] = temp_scaler.transform(validation_data[[column]])
        testing_data_normalized[column] = temp_scaler.transform(testing_data[[column]])

        #We willen de scaler van onze output_feature bewaren, aangezien we die moeten gebruiken om de output van de AI weer terug te schalen.
        if column == output_feature:
            scaler = temp_scaler
    
    return scaler, training_data_normalized, validation_data_normalized, testing_data_normalized


def save_dataframe(data: pd.DataFrame, path: str):
    data.to_csv(path, index=True)

def create_sequences(data_input, n_steps, future_hours, out_feature_index):
    """
    Functie die sequences maakt van onze data
    """
    data_input = data_input.values
    x, y = [], []
    for i in range(len(data_input) - n_steps - future_hours):
        end_ix = i + n_steps
        out_end_ix = end_ix + future_hours
        if out_end_ix > len(data_input):
            break
        seq_x, seq_y = data_input[i:end_ix, :], data_input[out_end_ix - 1, out_feature_index]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)

def test_model(model, x_testing, y_testing, scaler):
    #Model gebruiken om temperatuur te voorspellen
    predictions = model.predict(x_testing)
    #Output terugrekenen van scaled versie naar celsius
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    predictions_rounded = np.round(predictions_rescaled, output_decimals)
    #Hetzelfde doen we voor onze testing_data om de output van het model te vergelijken
    y_testing_rescaled = scaler.inverse_transform(y_testing.reshape(-1, 1))
    y_testing_rounded = np.round(y_testing_rescaled, output_decimals)

    #Verschil tussen prediction en daadwerkelijke waarde uitrekenen
    errors = predictions_rounded - y_testing_rounded
    #Dit verschil gebruiken om onze MAE te berekenen
    mae = np.mean(np.abs(errors))
    print(f"Mean Absolute Error: {mae}")

def main():
    #Path waar we ons dataframe van hourly_data.txt opslaan als csv
    data_path = './data/weather_data_api_daily.csv'
    combine_data()
    dataframe = create_pandas_frame()
    save_dataframe(dataframe, data_path)

    scaler, training, validation, testing = prepare_data_for_training(dataframe)

    x_train, y_train = create_sequences(training, n_steps, future_hours, output_feature_index)
    x_validation, y_validation = create_sequences(validation, n_steps, future_hours, output_feature_index)
    x_testing, y_testing = create_sequences(testing, n_steps, future_hours, output_feature_index)

    #Hier maken we het model dat we gaan trainen
    #We gebruiken als eerste laag 128 LSTM nodes, en aan het einde 1 dense node voor onze output_feature, temperatuur
    model = Sequential([
        LSTM(128,
             input_shape=(n_steps, x_train.shape[2])),
        Dense(1)
    ])

    #Compiler van het model instellen
    model.compile(optimizer='adam', loss='mse')

    #Callback die we gebruiken wanneer het model te inaccuraat wordt, hiermee proberen we overfitting te voorkomen
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=6,
        mode='min',
        restore_best_weights=True
    )

    #Hier start het trainen van het model.
    #epochs is het aantal generaties, batch_size bepaalt hoeveel data door het model gaat voordat de nodes geupdate worden
    history = model.fit(
        x_train, y_train,
        validation_data=(x_validation, y_validation),
        epochs=50,
        batch_size=2048,
        callbacks=[early_stop]
    )

    #Model en gebruikte scaler opslaan nadat we het model getrained hebben
    #De scaler slaan we op zodat we de predictions van het model ook weer terug kunnen rekenen naar graden celsius
    model.save("./model/weather_model.h5")
    dump(scaler, './model/scaler.pkl')
    
    test_model(model, x_testing, y_testing, scaler)

if __name__ == '__main__': 
    main()