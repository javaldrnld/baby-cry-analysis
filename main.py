import os
import select
import sys
import time
import wave

import joblib  # For loading scaler and encoder
import librosa
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model

from firebase_helper import upload_result
from vosk_offline import detect_human_voice_vosk  # Import VOSK offline detection

# Load environment variables
load_dotenv()

# Load pre-trained cry classification model
model = load_model("models/baby_classifier_model_v2.h5")

# Load saved StandardScaler and OneHotEncoder
scaler = joblib.load("models/scaler.pkl")  # Load saved scaler
encoder = joblib.load("models/encoder.pkl")  # Load saved encoder

# PyAudio configurations
FORMAT = "int16"
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
THRESHOLD = 12_000  #

COOLDOWN_PERIOD = 30

# Dictionary to count cry types
cry_count = {
    "belly_pain": 0,
    "burping": 0,
    "discomfort": 0,
    "hungry": 0,
    "tired": 0,
}


# Function to detect sound energy
def detect_sound(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    energy = np.sum(np.abs(audio_data))
    return energy > THRESHOLD


# Record 5-second audio
def record_audio():
    print("Recording...")
    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="int16", blocksize=CHUNK) as stream:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data, _ = stream.read(CHUNK)
                frames.append(data)

        if frames:
            frames = np.concatenate(frames, axis=0)
            wave_output_filename = "output.wav"
            write(wave_output_filename, RATE, frames)
            print("Recording saved:", wave_output_filename)
            return wave_output_filename
        else:
            print("No audio data captured.")
            return None

    except Exception as e:
        print(f"Error during recording: {e}")
        return None


# Feature extraction (same as used in training)
def extract_audio_features(data, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(data)), sr=sr).T, axis=0)
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

    # Map prediction to class names
    class_names = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
    return class_names[predicted_class]


# Real-time audio processing loop
def listen_for_baby_cry():
    print("Listening for baby cry...")
    last_detection_time = 0

    while True:
        current_time = time.time()

        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=FORMAT, blocksize=CHUNK) as stream:
            data, _ = stream.read(CHUNK)

            if detect_sound(data) and (current_time - last_detection_time) > COOLDOWN_PERIOD:
                audio_file = record_audio()

                if not audio_file:
                    print("No valid audio recorded, skipping classification.")
                    continue

                # Use VOSK for offline human voice detection
                if detect_human_voice_vosk(audio_file):
                    print("Human voice detected.")
                    upload_result("Human voice detected")
                else:
                    print("Baby cry detected.")
                    predicted_reason = classify_baby_cry(audio_file)
                    if predicted_reason and predicted_reason in cry_count:
                        cry_count[predicted_reason] += 1
                        print(f"Cry detected: {predicted_reason} ({cry_count[predicted_reason]} times)")
                        upload_result("Baby is crying", predicted_reason)
                    else:
                        print(f"Unknown cry reason detected.")

                last_detection_time = current_time
                print(f"Last detection time: {last_detection_time}")

        if (current_time - last_detection_time) <= COOLDOWN_PERIOD:
            remaining_cooldown = int(COOLDOWN_PERIOD - (current_time - last_detection_time))
            print(f"Cooldown period. Remaining time: {remaining_cooldown} seconds.")

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            user_input = input("Do you want to continue recording? (yes/no): ").lower()
            if user_input == "no":
                print("Stopping recording.")
                break

        time.sleep(0.1)


if __name__ == "__main__":
    listen_for_baby_cry()
