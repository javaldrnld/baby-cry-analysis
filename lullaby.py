import BlynkLib
import subprocess

# Initialize Blynk
blynk = BlynkLib.Blynk('hWRSpWldCWv-ha5aqmasPkbObCIvxNGX')

# Function to play audio
def play_audio():
    audio_file = "path_to_your_audio_file.wav" 
    try:
        subprocess.run(["aplay", audio_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while playing audio: {e}")

# Led control through V0 virtual pin
@blynk.on("V0")
def v0_write_handler(value):
    if int(value[0]) != 0:
        print('Playing Lullaby')
        play_audio()
    else:
        print('Not playing Lullaby')

@blynk.on("connected")
def blynk_connected():
    print("Alert: Hi! Raspberry Pi Connected to New Blynk2.0") 

while True:
    blynk.run()
