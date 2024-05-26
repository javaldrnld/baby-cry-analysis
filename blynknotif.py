import BlynkLib
import time

# Initialize Blynk
BLYNK_AUTH = 'YourAuthToken'
blynk = BlynkLib.Blynk(BLYNK_AUTH)
prediction = "tired"  # Change this value to test different conditions

def send_blynk_notification(message):
    blynk.notify(message)

# Function to handle different predictions
def handle_prediction(prediction):
    if prediction == "tired":
        send_blynk_notification("The baby is tired")
    elif prediction == "hungry":
        send_blynk_notification("The baby is hungry")
    elif prediction == "discomfort":
        send_blynk_notification("The baby is in discomfort")
    elif prediction == "burping":
        send_blynk_notification("The baby needs burping")
    elif prediction == "belly pain":
        send_blynk_notification("The baby has belly pain")

if __name__ == "__main__":
    # Connect to Blynk server and send the notification based on the prediction
    blynk.run()
    handle_prediction(prediction)

    # Keep the script running to maintain the Blynk connection
    while True:
        blynk.run()
        time.sleep(1)
