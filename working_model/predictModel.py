import os
import librosa
import numpy as np
import pickle


# Load the trained model, scaler, and label encoder
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


model_path = '/home/kotaro/Documents/baby-cry-analysis/notebooks/models/audio_augmented_model/rf_model.pkl'
scaler_path = '/home/kotaro/Documents/baby-cry-analysis/notebooks/models/audio_augmented_model/scaler_rf.pkl'
label_encoder_path = '/home/kotaro/Documents/baby-cry-analysis/notebooks/models/audio_augmented_model/label_encoder_rf.pkl'

trained_model = load_pickle(model_path)
scaler = load_pickle(scaler_path)
label_encoder = load_pickle(label_encoder_path)


# Define MFCC extraction
def mfcc(signal, sampling_rate=22050, frame_length=512, frame_shift=256, num_mfcc=13, n_mels=40, max_length=100):
    def preemphasis_filter(signal, alpha=0.97):
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])

    def frame_signal(signal, frame_length, frame_stride):
        signal_length = len(signal)
        frame_step = int(frame_stride)
        frame_length = int(frame_length)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        return frames

    def power_spectrum(frames, NFFT=512):
        frames *= np.hamming(frames.shape[1])
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
        return pow_frames

    def mel_filterbank(spectrum, num_filters=40, sampling_rate=22050, n_fft=512):
        mel_filterbank = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_filters)
        return np.dot(mel_filterbank, spectrum.T).T

    emphasized_signal = preemphasis_filter(signal)
    framed_signal = frame_signal(emphasized_signal, frame_length, frame_shift)
    spectrum = power_spectrum(framed_signal)
    mel_spectrum = mel_filterbank(spectrum, num_filters=n_mels, sampling_rate=sampling_rate, n_fft=512)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrum), n_mfcc=num_mfcc, n_mels=n_mels)

    if mfccs.shape[1] < max_length:
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs_padded = mfccs[:, :max_length]
    else:
        mfccs_padded = mfccs

    return mfccs_padded.flatten()


# Function to predict the label of an audio file
def predict_label(audio_path):
    signal, sr = librosa.load(audio_path, sr=None)
    mfcc_features = mfcc(signal, sampling_rate=sr)
    mfcc_features_flat = scaler.transform([mfcc_features])
    prediction = trained_model.predict(mfcc_features_flat)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict the label of an audio recording.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")

    args = parser.parse_args()
    label = predict_label(args.audio_path)
    print(f"Predicted label: {label}")
