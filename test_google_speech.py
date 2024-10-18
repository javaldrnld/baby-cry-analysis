import os
import sys

import pytest
from google.cloud import speech

from google_api_helper import detect_human_voice, get_transcript

# Add the current directory to Python's path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Test audio file path
TEST_AUDIO_FILE = "/home/bokuto/Documents/final_baby_elective/record.wav"  # Replace with the path to your test audio file


# Mock the response from Google Speech-to-Text API
class MockAlternative:
    transcript = "This is a test transcript"


class MockResult:
    alternatives = [MockAlternative()]


class MockResponse:
    results = [MockResult()]


# Test detect_human_voice function with a real audio file
def test_detect_human_voice():
    print("Testing detect_human_voice function...")

    # Ensure the Google Cloud Speech API is called correctly
    result = detect_human_voice(TEST_AUDIO_FILE)
    if result is None:
        print("No transcript detected or an error occurred.")
    else:
        print(f"Detected transcript: {result}")

    # You can also mock results for testing purposes without using Google API directly
    # For example, you can mock `detect_human_voice` response using `unittest.mock`


# Test get_transcript function with mock data
def test_get_transcript():
    print("\nTesting get_transcript function...")

    # Test with mock response
    transcript = get_transcript(MockResponse())
    assert (
        transcript == "This is a test transcript"
    ), f"Expected 'This is a test transcript', got {transcript}"

    # Test with empty response
    empty_response = speech.RecognizeResponse()
    assert get_transcript(empty_response) is None


if __name__ == "__main__":
    test_detect_human_voice()
    test_get_transcript()
    print("\nAll tests completed.")
