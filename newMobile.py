import BlynkLib
import RPi.GPIO as GPIO
import dht11
import time
import datetime
from BlynkTimer import BlynkTimer

# Initialize GPIO
GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

# Read data using pin 14
instance = dht11.DHT11(pin=14)

# Blynk authentication token
BLYNK_AUTH_TOKEN = 'jsfpl_ACPJIVDdkFJ9THE5PkCVvYR8FS'

# Initialize Blynk
blynk = BlynkLib.Blynk(BLYNK_AUTH_TOKEN)

# Create BlynkTimer Instance
timer = BlynkTimer()

# Function to sync the data from virtual pins
@blynk.on("connected")
def blynk_connected():
    print("Hi, You have Connected to New Blynk2.0")
    print(".......................................................")
    time.sleep(2)

# Function for collecting data from sensor & sending it to Server
def myData():
    result = instance.read()
    if result.is_valid():
        print("Last valid input: " + str(datetime.datetime.now()))
        print("Temperature: %-3.1f C" % result.temperature)
        print("Humidity: %-3.1f %%" % result.humidity)

        blynk.virtual_write(1, result.temperature)
        blynk.virtual_write(0, result.humidity)
        print("Values sent to New Blynk Server!")

# Set interval for data collection
timer.set_interval(2, myData)

# Run Blynk and Timer
while True:
    blynk.run()
    timer.run()
