import BlynkLib
import time
from BlynkTimer import BlynkTimer

# Initialize Blynk
BLYNK_AUTH = 'XIxKMdtkdcxn6RQUYeDRXqD8afPswzY3'
blynk = BlynkLib.Blynk(BLYNK_AUTH)

# Create BlynkTimer Instance
timer = BlynkTimer()

# Function to sync the data from virtual pins
@blynk.on("connected")
def blynk_connected():
    print("Hi, You have Connected to New Blynk2.0")
    print(".......................................................")
    time.sleep(2)
prediction = "burping"  

def send_blynk_notification(message):
    blynk.log_event(message)

# Function to handle different predictions
def handle_prediction(prediction):
    if prediction == "tired":
        print("The baby is tired")
        blynk.log_event("tired","The baby is tired")
    elif prediction == "hungry":
        print("The baby is hungry")
        blynk.log_event("hungry","The baby is hungry")
    elif prediction == "discomfort":
        print("The baby is in discomfort")
        blynk.log_event("discomfot","The baby is in discomfort")
    elif prediction == "burping":
        print("The baby needs burping")
        blynk.log_event("burping","The baby needs burping")
    elif prediction == "belly pain":
        print("The baby has belly pain")
        blynk.log_event("belly_pain","The baby has belly pain")
        

if __name__ == "__main__":
    # Connect to Blynk server and send the notification based on the prediction
    blynk.run()
    handle_prediction(prediction)

    # Keep the script running to maintain the Blynk connection
    while True:
        blynk.run()
        time.sleep(1)
