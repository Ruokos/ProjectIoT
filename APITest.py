import requests
import json
import pandas as pd
from datetime import datetime
def request_knmi_data(output_file: str):
    """
    Functie waarmee we de data van de KNMI api opvragen en opslaan in een .json file
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

    #Als we een response hebben, slaan we de content van deze response op in een .json
    if response.status_code == 200:
        with open(output_file, "wb") as file:
            file.write(response.content)
        print(f"Data saved in {output_file}")
    else:
        #Als we geen goede response hebben, zeggen we dit in terminal en stopt het programma
        print(f"No valid response from {url}, received code {response.status_code}")
        exit(1)

    #Hier zetten we de binary string van de KNMI response om in een JSON object
    content = response.content.decode("utf-8")
    json_data = json.loads(content)

    return json_data

def print_api_data() -> None:
    """
    Functie die de KNMI api gebruikt om data op te halen met request_knmi_data() functie om dit vervolgens uit te printen
    """
    json_data = request_knmi_data("knmi.json")
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

def create_pandas_frame():
    """
    Functie waarmee we opgevraagde data van de KNMI API opslaan in een pandas dataframe voor AI training
    """
    json_data = request_knmi_data("knmi.json")
    #Lijst waarin we dictionaries met dagelijkse data in gaan opslaan
    daily_data_list = []
    for day in json_data:
        #Datum osplaan als date object voor toekomstig sorteren van ons pandas dataframe
        date = day.get("date")[:-14]
        date = datetime.strptime(date, "%Y-%m-%d")
        #Temperatuur / 10 aangezien opgeslagen als 0.1 celsius
        temperature = day.get("TG") / 10
        #Luchtdruk delen door 10 mits de API luchtdruk teruggeeft
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
    daily_df = pd.DataFrame(daily_data_list)
    print(daily_df)

create_pandas_frame()