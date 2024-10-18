import io
import os

import librosa
import soundfile as sf
from dotenv import load_dotenv
from google.cloud import speech

# Load environment variables
load_dotenv()

# Debugging: Check if GOOGLE_APPLICATION_CREDENTIALS is loaded properly
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print(f"Google Credentials Path: {google_credentials}")  # Debugging statement

# Set Google Cloud credentials if the path exists
if google_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
else:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set. Check your .env file.")
# Initialize the Google Speech client
speech_client = speech.SpeechClient()


def detect_human_voice(audio_file):
    """Detect if the audio is human voice using Google Speech-to-Text API."""
    try:
        # Check the audio file properties (for debugging)
        check_audio_properties(audio_file)

        # Load the audio file and convert to mono
        data, sr = librosa.load(audio_file, sr=None, mono=True)

        # Save the mono version to a temporary file
        temp_mono_file = "temp_mono_audio.wav"
        sf.write(temp_mono_file, data, sr)

        # Read the content from the mono file for the Google API
        with io.open(temp_mono_file, "rb") as audio:
            content = audio.read()
            recognition_audio = speech.RecognitionAudio(content=content)

        # Set the language code (hardcoded for now to avoid .env issues)
        language_code = "en-US"  # Test hardcoding this value
        print(f"Using language code: {language_code}")  # Debug print

        # Configure the recognition request
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=language_code,  # Hardcoded language code for now
            audio_channel_count=1,  # Ensure the channel count is 1 (mono)
        )

        # Make the API call
        response = speech_client.recognize(config=config, audio=recognition_audio)
        return get_transcript(response)

    except Exception as e:
        print(f"Error in speech recognition: {str(e)}")
        # Print additional details if available
        if hasattr(e, "details"):
            print(f"Details: {e.details()}")
        return None


def get_transcript(response):
    """Extract transcript from the Google Speech API response."""
    if not response.results:
        return None

    first_result = response.results[0]
    if not first_result.alternatives:
        return None

    return first_result.alternatives[0].transcript.strip()


def check_audio_properties(audio_file):
    """Prints out the properties of the audio file for debugging."""
    try:
        f = sf.SoundFile(audio_file)
        print(f"Sample rate: {f.samplerate}")
        print(f"Channels: {f.channels}")
        print(f"Format: {f.format}")
        print(f"Subtype: {f.subtype}")
    except Exception as e:
        print(f"Error checking audio properties: {str(e)}")
