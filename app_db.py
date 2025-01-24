#!/bin/python3

# Project weather prediction
# Measuring station ESP32
# 18-01-2025


# settings
interval_duration = 3600 # seconds
database_file = "./db/sensor_readings.db"
database_table = "data"


import serial
import json
import sqlite3
from time import sleep
from datetime import datetime, timedelta


# function to send UART messages
def send_message(message):
    ser.write((str(message) + "\n").encode("utf-8"))
    print(f"sm: {message}")


# function to read UART messages
# also set up variable to use with the function
buffer = ""
def read_message(check=False):
    global buffer

    # read serial-buffer in "buffer" for more control
    if ser.in_waiting:
        buffer += ser.read(ser.in_waiting).decode("utf-8")

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
    

# function to insert rows in the database
def log_database(device_id, temperature, humidity, pressure):	
	conn=sqlite3.connect(database_file)
	curs=conn.cursor()
	curs.execute("INSERT INTO {} (device_id, temperature, humidity, pressure) VALUES (?, ?, ?, ?);".format(database_table), (device_id, temperature, humidity, pressure))
	conn.commit()
	conn.close()


# main function
def main():
    global ser

    # initialize UART connection with ESP32
    print("\n" + "connecting to ESP32...")
    ser = serial.Serial('/dev/ttyS0', baudrate=9600)
    ser.flush()
    while not read_message() == "connect?":
        sleep(1)
    send_message("ack")
    print("connection successful" + "\n")

    # initialize interval duration with ESP32
    print("synchronizing interval duration with ESP32...")
    while not read_message() == "interval?":
        sleep(1)
    send_message(interval_duration)
    print("synchronisation successful" + "\n")

    # set up a few parameters for determining the sleep duration later
    loop_start_time = datetime.now()
    loop_cycle_counter = 0

    while True:
        # count how many cycles the loop has been through
        loop_cycle_counter+=1

        # request and store the sensor data
        send_message("data?")
        while not read_message(check=True):
            sleep(1)
        data = json.loads(read_message())


        # access individual measurements
        device_id = str(data.get('id'))
        temperature = str(data.get('temp'))
        humidity = str(data.get('hum'))
        pressure = str(data.get('pres'))


        # store measurements in database
        log_database(device_id, temperature, humidity, pressure)

        # calculate and sleep for the set interval time
        loop_new_time = loop_start_time + timedelta(seconds=(loop_cycle_counter * interval_duration))
        loop_sleep_duration = (loop_new_time - datetime.now()).total_seconds()
        if loop_sleep_duration < 0: loop_sleep_duration = 0
        sleep(loop_sleep_duration)


if __name__ == '__main__': 
    main()


# Close the serial connection
ser.close()