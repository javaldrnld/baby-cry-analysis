import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, TimeStretch
# Compose -> Augmentation Chain

from helper import  _plot_signal_and_augmented_signal

augment_raw_audio = Compose([
    AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=1),
    PitchShift(min_semitones=-8, max_semitones=10, p=1),
    HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=1),
    TimeStretch(min_rate=0.1, max_rate=0.5, p=1)
])

if __name__ == "__main__":
    signal, sr = librosa.load("06c4cfa2-7fa6-4fda-91a1-ea186a4acc64-1430029237378-1.7-f-26-ti.wav")
    augmented_signal = augment_raw_audio(signal, sr)
    sf.write("augmented_raw_audio.wav", augmented_signal, sr)
    _plot_signal_and_augmented_signal(signal, augmented_signal, sr)