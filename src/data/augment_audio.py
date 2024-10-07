from src.data.make_dataset import split_audio


def augment_dataset(raw_audio_full):
    """
    Augments a dataset of raw audio files by splitting each audio file into parts.

    Args:
        raw_audio_full (dict): A dictionary where keys are paths to raw audio files and values are their corresponding labels.

    Returns:
        dict: A dictionary where keys are parts of the original audio files and values are their corresponding labels.
    """

    raw_audio_augmented = {}

    for audio_file, label in raw_audio_full.items():
        parts = split_audio(audio_file)
        for part in parts:
            raw_audio_augmented[part] = label
    return raw_audio_augmented


