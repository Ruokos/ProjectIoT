import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os


def combine_data():
    folder_path = "C:\\Github Repositories\\ProjectIoT\\data\\uurmetingen"
    output = "C:\\Github Repositories\\ProjectIoT\\data\\hourly_data.txt"

    text_files = [file for file in os.listdir(folder_path)]

    with open(output, "w") as outfile:
        for txt_file in text_files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, "r") as infile:
                outfile.write(infile.read())

def load_data():
    data = "C:\\Github Repositories\\ProjectIoT\\data\\hourly_data.txt"
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
                "pressure": pressure,
                'humidity': humidity,
                'temperature': temperature

            }
            extracted_data.append(data)
    
    hourly_dataframe = pd.DataFrame(extracted_data)
    print(hourly_dataframe)

combine_data()
load_data()