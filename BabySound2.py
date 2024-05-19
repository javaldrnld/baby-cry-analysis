import pyaudio
import wave
import numpy as np


def record_audio(filename, duration=5, threshold=5000):
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []
    
    try:
        print("Listening for loud noise...")
        while True:
            data = stream.read(1024)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.max(audio_data) > threshold:
                print("Loud noise detected! Recording...")
                break
    except KeyboardInterrupt:
        pass
    
    print("Recording...")
    for _ in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    print("Recording stopped.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    sound_file = wave.open(filename, "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(frames))
    sound_file.close()


def saveaudio():
    if __name__ == "__main__":
        filenumber = 0
        while True:
            if filenumber == 5: 
             filenumber = 0
            else:
                filenumber += 1
        
            filename = f"recording_{filenumber}.wav"
            record_audio(filename)


saveaudio()