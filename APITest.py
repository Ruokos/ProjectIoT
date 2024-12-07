import requests

url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"

data = {
    "start": 20000101,
    "end": 20240101,
    "stns": "ALL",
    "fmt": "xml",
    "stns": "277"
    }

response = requests.post(url, data=data, verify=False)

if response.status_code == 200:
    output_file = "knmi.xml"
    with open(output_file, "wb") as file:
        file.write(response.content)
    print(f"Data saved in {output_file}")
else:
    print(f"No valid response from {url}")