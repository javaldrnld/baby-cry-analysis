import os
import wave

import librosa
import soundfile as sf
from vosk import KaldiRecognizer, Model


def detect_human_voice_vosk(audio_file, model_path="vosk-model-small-en-us-0.15"):
    """Detect if the audio contains human voice using the VOSK API offline."""

    # Load the VOSK model
    if not os.path.exists(model_path):
        raise ValueError(f"Model path '{model_path}' does not exist.")

    model = Model(model_path)

    try:
        # Check and preprocess the audio file
        audio_file = preprocess_audio(audio_file)

        # Open the processed audio file
        with wave.open(audio_file, "rb") as wf:
            recognizer = KaldiRecognizer(model, wf.getframerate())
            results = []
            while True:
                data = wf.readframes(4000)  # Read 4000 frames at a time
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    results.append(recognizer.Result())

            # Final result after the loop
            results.append(recognizer.FinalResult())
            return get_vosk_transcript(results)

    except Exception as e:
        print(f"Error in VOSK speech recognition: {str(e)}")
        return None


def preprocess_audio(audio_file, target_sr=16000):
    """Ensure the audio is in the correct format (16kHz mono WAV)."""
    with sf.SoundFile(audio_file) as f:
        if f.samplerate != target_sr or f.channels != 1:
            print("Resampling and converting to mono...")
            # Load and resample the audio
            data, sr = librosa.load(audio_file, sr=target_sr, mono=True)
            temp_audio = "temp_16khz_mono.wav"
            sf.write(temp_audio, data, target_sr)
            return temp_audio
        return audio_file  # Return original if already compliant


def get_vosk_transcript(results):
    """Extract and compile the transcript from VOSK recognition results."""
    transcript = ""
    for result_json in results:
        if result_json.strip():
            result = eval(result_json)  # Parse the JSON result
            if "text" in result:
                transcript += result["text"] + " "
    return transcript.strip() if transcript else None


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


# Example Usage:
# transcript = detect_human_voice_vosk("record.wav")
# print(f"Transcript: {transcript}")
