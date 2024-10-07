import io
import time
import wave

import firebase_admin
import numpy as np
import pyaudio
from firebase_admin import credentials, db
from google.cloud import speech
from google.oauth2 import service_account

# Firebase and Google Cloud Setup
cred = credentials.Certificate("cry_firebase.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://baby-9e20f-default-rtdb.asia-southeast1.firebasedatabase.app/"
    },
)
ref = db.reference()

client_file = "demo_speech.json"
credentials = service_account.Credentials.from_service_account_file(client_file)
speech_client = speech.SpeechClient(credentials=credentials)


import time
from datetime import datetime

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
THRESHOLD = 500

# Initialize PyAudio
paudio = pyaudio.PyAudio()


# Firebase function to upload result
def upload_result(result):
    ref = db.reference()

    # Get the current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Update the database
    updates = {
        "baby_cry_status/status": "cry" if result == "Baby is crying" else "no cry",
    }

    ref.update(updates)

    # update the list fields


# Function to process audio data
def detect_sound(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    energy = np.sum(np.abs(audio_data))
    return energy > THRESHOLD


# Google Cloud Speech-to-Text API
def detect_human_voice(audio_file):
    with io.open(audio_file, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
    )

    response = speech_client.recognize(config=config, audio=audio)
    transcript = get_transcript(response)

    return transcript is not None


# Extract transcript
def get_transcript(response):
    if not response.results:
        return None  # No transcript

    first_result = response.results[0]
    if not first_result.alternatives:
        return None  # No alternatives, no transcript

    transcript = first_result.alternatives[0].transcript.strip()
    return transcript if transcript else None


# Function to record audio
def record_audio(stream):
    print("Recording...")
    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    wave_output_filename = "output.wav"
    wf = wave.open(wave_output_filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(paudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    return wave_output_filename


# Main real-time audio proecssing loop
def listen_for_baby_cry():
    stream = paudio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Listening for baby cry...")

    while True:
        ref.update({"RecordStatus": "recording"})
        data = stream.read(CHUNK, exception_on_overflow=False)

        if detect_sound(data):
            # Record 5-second audio
            audio_file = record_audio(stream)

            # Use GOogle cloude API
            if detect_human_voice(audio_file):
                print("Human voice...")
                upload_result("Human voice detected")
            else:
                print("Baby cry...")
                upload_result("Baby is crying")

            while True:
                user_input = input(
                    "Do you wan tto continue recording? (yes/no): "
                ).lower()
                if user_input in ["yes", "no"]:
                    break
                print("Invalid input")

            if user_input == "no":
                print("Stopping recording. Goodbye")
                stream.stop_stream()
                stream.close()
                paudio.terminate()
                return


if __name__ == "__main__":
    listen_for_baby_cry()
