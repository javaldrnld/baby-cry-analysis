import wave
from unittest.mock import MagicMock, patch

import librosa
import numpy as np
import pyaudio
import pytest

from main import classify_baby_cry, detect_sound, listen_for_baby_cry, record_audio

# Define some test constants
TEST_AUDIO_FILE = "test_audio.wav"


# Mock PyAudio stream
@pytest.fixture
def mock_pyaudio_stream():
    stream = MagicMock()
    stream.read = MagicMock(
        return_value=np.random.rand(1024).astype(np.int16).tobytes()
    )  # Random data simulating audio
    return stream


# Mock the detect_sound function
@patch("main.detect_sound", return_value=True)
@patch("main.record_audio", return_value=TEST_AUDIO_FILE)
@patch("main.detect_human_voice", return_value=False)  # Simulating baby cry detected
@patch(
    "main.classify_baby_cry", return_value="hungry"
)  # Simulating model predicting 'hungry'
@patch("main.upload_result")
def test_listen_for_baby_cry(
    mock_upload_result,
    mock_classify_baby_cry,
    mock_detect_human_voice,
    mock_record_audio,
    mock_detect_sound,
    mock_pyaudio_stream,
):
    # Patch the PyAudio initialization and stream
    with patch("main.paudio.open", return_value=mock_pyaudio_stream):
        with patch(
            "builtins.input", side_effect=["no"]
        ):  # Simulate user input 'no' to exit loop
            listen_for_baby_cry()

    # Assertions
    mock_detect_sound.assert_called()  # Check that detect_sound was called
    mock_record_audio.assert_called_once()  # Ensure that recording happened
    mock_detect_human_voice.assert_called_once_with(
        TEST_AUDIO_FILE
    )  # Ensure Google API function was called
    mock_classify_baby_cry.assert_called_once_with(
        TEST_AUDIO_FILE
    )  # Ensure classification was done
    mock_upload_result.assert_called_once_with(
        "Baby is crying", "hungry"
    )  # Ensure the result was uploaded


# Mock the classification function (no real model inference)
@patch("main.classify_baby_cry", return_value="hungry")
def test_classify_baby_cry(mock_classify_baby_cry):
    result = classify_baby_cry(TEST_AUDIO_FILE)
    assert result == "hungry"


# Test sound detection function
def test_detect_sound():
    data = np.random.rand(1024).astype(np.int16).tobytes()  # Random sound data
    assert detect_sound(data) == True


# Test record_audio function with mocked stream
@patch("wave.open")
def test_record_audio(mock_wave_open, mock_pyaudio_stream):
    mock_wave_obj = MagicMock()
    mock_wave_open.return_value = mock_wave_obj

    result = record_audio(mock_pyaudio_stream)
    assert result == "output.wav"
    mock_wave_open.assert_called_once_with("output.wav", "wb")
    mock_wave_obj.writeframes.assert_called()  # Ensure audio frames were written
