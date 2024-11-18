# Calibration function
import librosa
import matplotlib.pyplot as plt
import numpy as np


def calibrate_threshold(audio_files, sample_rate=44100):
    """
    Calibrate an appropriate sound energy threshold for distinguishing baby cries from noise.

    Args:
        audio_files (list of str): List of file paths to audio samples.
                                   Include both baby cries and background noises.
        sample_rate (int): Sample rate for loading audio files. Default is 44100 Hz.

    Returns:
        float: Suggested threshold based on the average energy of cry samples.
    """
    energies = []
    labels = []

    for file in audio_files:
        print(f"Processing {file}...")
        # Load the audio file
        data, _ = librosa.load(file, sr=sample_rate)

        # Calculate energy
        energy = np.sum(np.abs(data))
        energies.append(energy)

        # Label based on file name (update logic for labeling if necessary)
        labels.append("cry" if "cry" in file.lower() else "noise")

    # Separate cries and noise energies for averaging
    cry_energies = [e for e, label in zip(energies, labels) if label == "cry"]
    noise_energies = [e for e, label in zip(energies, labels) if label == "noise"]

    # Calculate average energy for cries and noise
    avg_cry_energy = np.mean(cry_energies) if cry_energies else 0
    avg_noise_energy = np.mean(noise_energies) if noise_energies else 0

    # Suggested threshold (you may adjust based on your environment)
    suggested_threshold = (avg_cry_energy + avg_noise_energy) / 2

    # Plot the energy levels
    plt.scatter(
        range(len(energies)), energies, c=["r" if label == "cry" else "b" for label in labels], label="Energy levels"
    )
    plt.axhline(y=suggested_threshold, color="g", linestyle="--", label="Suggested Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Energy Level")
    plt.title("Energy Levels for Baby Cries vs Noise")
    plt.legend()
    plt.show()

    print(f"Average Cry Energy: {avg_cry_energy}")
    print(f"Average Noise Energy: {avg_noise_energy}")
    print(f"Suggested Threshold: {suggested_threshold}")

    return suggested_threshold


if __name__ == "__main__":
    # Replace with the actual file paths for calibration

    audio_files = [
        "/home/untitled/Documents/baby-cry-analysis/test_cry/cry1.wav",
        "/home/untitled/Documents/baby-cry-analysis/test_cry/cry2.wav",
        "/home/untitled/Documents/baby-cry-analysis/test_cry/cry3.wav",
    ]

    # Run the calibration
    threshold = calibrate_threshold(audio_files)
    print(f"Calibrated Threshold: {threshold}")
