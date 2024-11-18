import time

import numpy as np
import sounddevice as sd

# Sounddevice configurations
FORMAT = "int16"
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 10000000000000  # Adjust threshold based on ambient noise level


# Function to detect sound energy
def detect_sound(data):
    energy = np.sum(np.abs(data))
    return energy > THRESHOLD


# Real-time audio test loop
def test_real_time_audio():
    print("Testing real-time audio input...")
    print("Speak or make noise to test the energy threshold.")
    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=FORMAT, blocksize=CHUNK) as stream:
            while True:
                # Read audio data
                data, _ = stream.read(CHUNK)
                data = np.frombuffer(data, dtype=np.int16)  # Convert audio to NumPy array

                # Detect if sound energy exceeds the threshold
                if detect_sound(data):
                    print("Sound detected above threshold!")
                else:
                    print("Sound below threshold.")

                time.sleep(0.1)  # Short delay to simulate processing

    except KeyboardInterrupt:
        print("\nTest interrupted. Exiting.")


if __name__ == "__main__":
    test_real_time_audio()
