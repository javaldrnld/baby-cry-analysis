import librosa


def load_and_process_audio(audio_path, sr=None):
    signal, sr = librosa.load(audio_path, sr=sr)
    return signal, sr


# You can add more utility functions here
