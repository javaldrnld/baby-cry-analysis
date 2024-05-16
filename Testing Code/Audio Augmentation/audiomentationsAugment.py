import os
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, TimeStretch
# Compose -> Augmentation Chain

from helper import _plot_signal_and_augmented_signal

augment_raw_audio = Compose([
    AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.5),
    PitchShift(min_semitones=-8, max_semitones=10, p=1),
    HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=1),
    TimeStretch(min_rate=0.1, max_rate=0.5, p=1)
])


def augment_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                signal, sr = librosa.load(file_path)
                augmented_signal = augment_raw_audio(signal, sr)

                # Create output directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Prepend 'audio_augmented' to the file name
                augmented_filename = f"audio_augmented_{file}"
                output_file_path = os.path.join(output_folder, augmented_filename)

                sf.write(output_file_path, augmented_signal, sr)
                print(f"Augmented {file_path} and save to {output_file_path}")


if __name__ == "__main__":
    input_directory = "/home/kotaro/Documents/baby-cry-analysis/data/raw"
    output_directory = "/home/kotaro/Documents/baby-cry-analysis/data/raw/Augmented_Audio"
    augment_audio_files(input_directory, output_directory)
