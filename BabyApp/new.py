import time
import signal
import subprocess
import firebase_admin
from firebase_admin import credentials, db as firebase_db

#import audio # name nung file prediction file


# Firebase Admin credentials setup
cred = credentials.Certificate("C:/Users/Mike/Downloads/baby.json")

# Update with your service account JSON file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://baby-9e20f-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Global variable to store the process handle for audio
# audio_process = None


# def play_lullaby():
#     global audio_process
#     if audio_process is None or audio_process.poll() is not None:
#         audio_file = "lullaby.wav"
#         try:
#             audio_process = subprocess.Popen(["aplay", audio_file])
#         except subprocess.CalledProcessError as e:
#             print(f"Error occurred while playing lullaby: {e}")

# def stop_audio():
#     global audio_process
#     if audio_process:
#         audio_process.terminate()
#         audio_process = None

def read_record_status():
    record_status = firebase_db.reference('RecordStatus').get()
    return record_status if record_status else ""

def read_lullaby_status():
    lullaby_status = firebase_db.reference('lullabyStatus').get()
    return lullaby_status if lullaby_status else ""

if __name__ == "__main__":
    #signal.signal(signal.SIGINT, audio.stop_recording)
    print("Press Ctrl+C to stop.")

    while True:
        record_status = read_record_status()
        lullaby_status = read_lullaby_status()

        if record_status == "recording":
            prediction = "hungry"
            #prediction = audio.record_audio()
            try:
                # Send 'prediction' to Firebase under a node named "Predictions"
                firebase_db.reference('Predictions').set(prediction)
                print("Prediction sent to Firebase:", prediction)
            except Exception as e:
                print(f"Error sending prediction to Firebase: {e}")

        else:
            print("Recording not allowed. RecordStatus:", record_status)

        if lullaby_status == "playing":
            print("playing")
            #play_lullaby()
        elif lullaby_status == "paused":
            print("not playing")
            #stop_audio()

        time.sleep(1)  # Adjust sleep time as needed



