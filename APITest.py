import requests
import json

#URL voor KNMI api
url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"

#Configuratie voor opvragen data
data = {
    "start": 20000101,
    "end": 20240601,
    "stns": "ALL",
    "fmt": "json",
    "stns": "277"
    }

#Response van de API
response = requests.post(url, data=data, verify=False)

#Als we een response hebben, slaan we de content van deze response op in een .json
if response.status_code == 200:
    output_file = "knmi.json"
    with open(output_file, "wb") as file:
        file.write(response.content)
    print(f"Data saved in {output_file}")
else:
    #Als we geen goede response hebben, zeggen we dit in terminal en stopt het programma
    print(f"No valid response from {url}")
    exit(1)

#Hier zetten we de binary string van de KNMI response om in een JSON object
content = response.content.decode("utf-8")
json_data = json.loads(content)

#For loop waarin we kunnen iteraten over alle dagen die we opgevraagd hebben
for day in json_data:
    date = day.get("date")
    print(date)
