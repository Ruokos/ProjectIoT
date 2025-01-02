import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

numeric_columns = ['tempC', 'pressure', 'humidity']
#Waarden die we nodig hebben voor het maken van sequences om het model te trainen
output_feature = "tempC"
output_feature_index = numeric_columns.index(output_feature)
#We willen 1 dag in de toekomst de temperatuur voorspellen
future_days = 1
#Hiervoor gebruiken we n_steps afgelopen dagen 
n_steps = 365
#Om te voorspellen gebruiken we 3 waarden, temperatuur, luchtdruk en luchtvochtigheid
n_inputs = 3

def request_knmi_data():
    """
    Functie waarmee we de data van de KNMI api opvragen
    Functie geeft data ook terug als json object om in andere functies te gebruiken
    """
    #URL voor KNMI api
    url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"

    #Configuratie voor opvragen data
    data = {
        "start": 20000101,
        "end": 20240601,
        "stns": "277",
        "fmt": "json",
        }

    #Response van de API
    response = requests.post(url, data=data, verify=False)

    #Hier zetten we de binary string van de KNMI response om in een JSON object
    if response.status_code == 200:
        content = response.content.decode("utf-8")
        json_data = json.loads(content)

        return json_data
    
    #Als we geen goede response hebben, zeggen we dit in terminal en stopt het programma
    else:
        print(f"No valid response from {url}, received code {response.status_code}")
        exit(1)

def print_api_data(json_data) -> None:
    """
    Functie die de KNMI api gebruikt om data op te halen met request_knmi_data() functie om dit vervolgens uit te printen
    """
    #For loop waarin we kunnen iteraten over alle dagen die we opgevraagd hebben
    for day in json_data:
        #[:-14] om onnodige timestamp weg te halen, dit is namelijk toch overal 00:00:00
        date = day.get("date")[:-14]
        #Temperatuur wordt gegeven in 0.1 celsius, vandaar / 10
        average_temperature = day.get("TG") / 10
        #Luchtdruk moet ook / 10, maar huidige station heeft geen luchtdrukwaarden dus voor nu niks om geen error te veroorzaken
        average_pressure = day.get("PG") 
        average_humidity = day.get("UG")
        #Hier printen we het uit, dit moet uiteindelijk in een pandas dataframe worden gestopt voor AI training
        print(f"Datum: {date}, temperatuur: {average_temperature}, luchtdruk: {average_pressure}, luchtvochtigheid: {average_humidity}")

def create_pandas_frame(json_data) -> pd.DataFrame:
    """
    Functie waarmee we opgevraagde data van de KNMI API opslaan in een pandas dataframe
    """
    #Lijst waarin we dictionaries met dagelijkse data in gaan opslaan
    daily_data_list = []
    #For loop waarin we van alle dagen de informatie opslaan in een dictionary
    for day in json_data:
        #Datum osplaan als date object voor toekomstig sorteren van ons pandas dataframe
        date = day.get("date")[:-14]
        date = datetime.strptime(date, "%Y-%m-%d")
        #Temperatuur / 10 aangezien opgeslagen als 0.1 celsius
        temperature = day.get("TG") / 10
        #Luchtdruk delen door 10 mits de API luchtdruk waarden heeft
        pressure = day.get("PG")
        if pressure is not None:
            pressure /= 10
        humidity = day.get("UG")
        
        #Data van een dag opslaan in een dictionary
        daily_data = {
            'date': date,
            'tempC': temperature,
            'pressure': pressure,
            'humidity': humidity
        }
        daily_data_list.append(daily_data)
    
    #Dataframe maken van onze lijst van dictionaries
    daily_df = pd.DataFrame(daily_data_list)

    #Zeker weten dat we alle numerieke dingen ook daadwerkelijk opslaan als nummer. Als iets niet als nummer opgeslagen kan worden, bijvoorbeeld None, wordt dit NaN.
    daily_df[numeric_columns] = daily_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    #Als index van ons DataFrame gebruiken we de datum
    daily_df.set_index('date', inplace=True)

    #Zeker maken dat we daadwerkelijk alles op datum gesorteerd hebben
    daily_df.sort_values(by='date', inplace=True)
    
    return daily_df

def prepare_data_for_training(data: pd.DataFrame):
    """
    This function splits up the dataset for training, validation and testing and also normalizes all values
    """
    split = 0.8
    training_size = int(len(data) * split)
    validation_size = int(len(data) * (1-split)//2)
    testing_size = len(data) - training_size - validation_size

    training_data = data[:training_size]
    validation_data = data[training_size:training_size + validation_size]
    testing_data = data[-testing_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))

    training_data_normalized = scaler.fit_transform(training_data)
    validation_data_normalized = scaler.fit_transform(validation_data)
    testing_data_normalized = scaler.fit_transform(testing_data)

    return training_data_normalized, validation_data_normalized, testing_data_normalized


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
    

def main():
    json_data = request_knmi_data()
    dataframe = create_pandas_frame(json_data)
    save_dataframe(dataframe, './data/weather_data_api_daily.csv')

    training, validation, testing = prepare_data_for_training(dataframe)

    x_train, y_train = create_sequences(training, n_steps, future_days, output_feature_index)
    x_validation, y_validation = create_sequences(validation, n_steps, future_days, output_feature_index)
    x_testing, y_testing = create_sequences(testing, n_steps, future_days, output_feature_index)


main()
