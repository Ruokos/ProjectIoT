# Project weather prediction
# Measuring station ESP32
# 23-01-2025
# main.py file


# settings
interval_tolerance = 30 # sec
red_LED_pin = 23
green_LED_pin = 19
SCL_pin = 22
SDA_pin = 21


import json
import BME280
from time import sleep
from machine import UART, Pin, I2C


# function to send UART messages
def send_message(message):
    uart.write((str(message) + "\n").encode("utf-8"))
    print(f"sm: {message}")
    
    
# function to read UART messages
# also set up variable to use in the function
buffer = ""
def read_message(check=False):
    global buffer
    
    # read serial-buffer in "buffer" for more control
    if uart.any():
        buffer += uart.read().decode("utf-8")
        
    # read new message
    if ("\n" in buffer) and (not check):
        newline_pos = buffer.find("\n") + 1
        message = buffer[:newline_pos].strip()
        buffer = buffer[newline_pos:]
        print(f"rm: {message}") # {repr(message)}
        return message
    
    # only check if there is a new message
    elif ("\n" in buffer) and check:
        return True
    
    # if there is no new message
    else:
        return None


# initialize LEDs
ledGrn = Pin(green_LED_pin, Pin.OUT)
ledRd = Pin(red_LED_pin, Pin.OUT)


# 'initialization' state LEDs
ledGrn.value(1)
ledRd.value(1)


# initialize I2C connection with BME280
print("\n" + "connecting to BME280...")
i2c = I2C(scl=Pin(SCL_pin), sda=Pin(SDA_pin), freq=10000)
bme = BME280.BME280(i2c=i2c)
print("connection successful" + "\n")


# initialize UART connection with Raspberry PI
print("connecting to Raspberry PI...")
uart = UART(2, 9600)
uart.flush()
send_message("connect?")
while not read_message() == "ack":
    sleep(1)
print("connection successful" + "\n")


# initialize interval duration with Raspberry PI
print("synchronizing interval duration with Raspberry PI...")
send_message("interval?")
while not read_message(check=True):
    sleep(1)
interval_duration_rasp = int(read_message())
interval_duration_esp = interval_duration_rasp - interval_tolerance
if interval_duration_esp < 0: interval_duration_esp = 0
print("synchronisation successful")
print(f"interval duration rasp: {interval_duration_rasp}")
print(f"interval duration esp: {interval_duration_esp}" + "\n")


# 'operating' state LEDs
ledGrn.value(1)
ledRd.value(0)


while True:
    try:
        # wait until data request from Rasbperry PI is received.
        # If this takes too long there is most likely an error.
        loop_cycle_counter = 0
        while not read_message() == "data?":
            if loop_cycle_counter > (2*interval_tolerance):
                raise Exception("UART error")
            loop_cycle_counter+=1
            sleep(1)
        
        # read BME280 data
        temp = bme.temperature
        hum = bme.humidity
        pres = bme.pressure
            
        # create BME280 JSON data, seperating units, and rounding
        data = {
            "id": "tuin",
            "temp": round(float(temp.split("C")[0]), 1),
            "hum": round(float(hum.split("%")[0]), 1),
            "pres": round(float(pres.split("hPa")[0]), 1)
        }
        JSON_data = json.dumps(data)
        
        # send BME280 JSON data
        send_message(JSON_data)
        
        # sleep for calculated interval duration
        sleep(interval_duration_esp)
        
    except Exception as e:
        # 'error' state LEDs
        ledGrn.value(0)
        ledRd.value(1)
        
        print(f"\nerror: {e}")
        send_message(f"error: {e}")
        
        break