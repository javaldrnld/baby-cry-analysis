import random

import librosa
import numpy as np
import soundfile as sf
from helper import _plot_signal_and_augmented_signal


# adding white noise
def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size) # Can add other noise and play with noise factor
    augmented_signal = signal + noise * noise_factor
    return augmented_signal


# Time Stretch
def time_stretch(signal, stretch_rate):
    return librosa.effects.time_stretch(signal, rate=stretch_rate)


# Pitch Scaling
def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=num_semitones)


# Polarity Inversion
def invert_polarity(signal):
    return signal * -1


# Random Gain
def random_gain(signal, min_gain_factor, max_gain_factor):
    gain_factor = random.uniform(min_gain_factor, max_gain_factor)
    return signal * gain_factor

if __name__ == "__main__":
    signal, sr = librosa.load("scale.wav")
    augmented_signal = random_gain(signal, 2, 3)
    sf.write("augmented.wav", augmented_signal, sr)
    _plot_signal_and_augmented_signal(signal, augmented_signal, sr)


