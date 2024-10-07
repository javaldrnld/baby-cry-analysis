import os

from pydub import AudioSegment


def load_raw_audio(directories, base_path):
    """
    Loads raw audio files from specified directories and stores them in a dictionary.
    Args:
        directories (list): A list of directory names to search for audio files.
        base_path (str): The base path where the directories are located.
    Returns:
        dict: A dictionary where the keys are the full paths to the audio files and the values are the directory names.
    """

    raw_audio_full = {}

    # Load all the audio files into a dictionary
    for directory in directories:
        path = "/home/bokuto/Documents/baby-cry-analysis/data/clean_raw"
        for filename in os.listdir(path):
            if filename.endswith(".wav"):
                raw_audio_full[os.path.join(path, filename)] = directory
    return raw_audio_full


# To create second dataset
def split_audio(audio_path, num_parts=10):
    """
    Splits an audio file into a specified number of equal parts.

    Args:
        audio_path (str): The file path to the audio file to be split.
        num_parts (int, optional): The number of parts to split the audio into. Default is 10.

    Returns:
        list: A list of AudioSegment objects, each representing a part of the original audio.
    """

    sound = AudioSegment.from_file(audio_path, format="wav")
    duration = len(sound)
    part_duration = duration // num_parts
    parts = [
        sound[i * part_duration : (i + 1) * part_duration] for i in range(num_parts)
    ]
    return parts
