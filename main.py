import os
import select
import sys
import time
import wave

import joblib  # For loading scaler and encoder
import librosa
import numpy as np
import pyaudio
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from firebase_helper import upload_result
from google_api_helper import detect_human_voice

# Load environment variables
load_dotenv()

# Load pre-trained cry classification model
model = load_model("models/baby_classifier_model_v2.h5")

# Load saved StandardScaler and OneHotEncoder
scaler = joblib.load("models/scaler.pkl")  # Load saved scaler
encoder = joblib.load("models/encoder.pkl")  # Load saved encoder

# PyAudio configurations
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
THRESHOLD = 500

COOLDOWN_PERIOD = 30

# Initialize PyAudio
paudio = pyaudio.PyAudio()


# Function to detect sound energy
def detect_sound(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    energy = np.sum(np.abs(audio_data))
    return energy > THRESHOLD


# Record 5-second audio
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


# Feature extraction (same as used in training)
def extract_audio_features(data, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    chroma_stft = np.mean(
        librosa.feature.chroma_stft(S=np.abs(librosa.stft(data)), sr=sr).T, axis=0
    )
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    return np.hstack([zcr, chroma_stft, mfcc, rms, mel])


# Classify baby cry using the loaded model
def classify_baby_cry(audio_file):
    # Load and extract features from the audio file
    data, sr = librosa.load(audio_file, duration=2.5, offset=0.6)
    features = extract_audio_features(data, sr)

    if features.size == 0:
        return "Unknown"

    # Use the loaded scaler to standardize the features
    features = scaler.transform([features])  # Standardize features
    features = np.expand_dims(features, axis=2)  # Reshape for CNN input

    # Predict the class using the loaded model
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    # Map prediction to class names (you may use encoder.inverse_transform if you saved it)
    class_names = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
    return class_names[predicted_class]


# Real-time audio processing loop
def listen_for_baby_cry():
    stream = paudio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Listening for baby cry...")
    last_detection_time = 0

    while True:
        current_time = time.time()
        data = stream.read(CHUNK, exception_on_overflow=False)

        if (
            detect_sound(data)
            and (current_time - last_detection_time) > COOLDOWN_PERIOD
        ):
            audio_file = record_audio(stream)

            if detect_human_voice(audio_file):
                print("Human voice detected.")
                upload_result("Human voice detected")
            else:
                print("Baby cry detected.")
                predicted_reason = classify_baby_cry(audio_file)
                upload_result("Baby is crying", predicted_reason)

            last_detection_time = current_time
            print(f"Last detection time: {last_detection_time}")

        if (current_time - last_detection_time) <= COOLDOWN_PERIOD:
            remaining_cooldown = int(
                COOLDOWN_PERIOD - (current_time - last_detection_time)
            )
            print(f"Cooldown period. Remaining time: {remaining_cooldown} seconds.")

        # Non-blocking input check
        if select.select([sys.stdin], [], [], 0.0)[0]:
            user_input = input("Do you want to continue recording? (yes/no): ").lower()
            if user_input == "no":
                print("Stopping recording. Goodbye.")
                stream.stop_stream()
                stream.close()
                paudio.terminate()
                break
        time.sleep(0.1)


if __name__ == "__main__":
    listen_for_baby_cry()
