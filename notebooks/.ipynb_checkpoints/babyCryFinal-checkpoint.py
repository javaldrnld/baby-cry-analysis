#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import librosa
import wave
import math
import uuid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from pydub import AudioSegment


# In[6]:


# Define raw audio dictionary
raw_audio_full = {}
raw_audio_augmented = {}

# Loop through directioris and label audio files
directories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
for directory in directories:
    path = '../data/donateacry_corpus/' + directory
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            raw_audio_full[os.path.join(path, filename)] = directory


# In[8]:


# Split the audio file to create second dataset
def split_audio(audio_path, num_parts=10):
    sound = AudioSegment.from_file(audio_path, format="wav")
    duration = len(sound)
    part_duration = duration // num_parts
    parts = [sound[i*part_duration:(i+1)*part_duration] for i in range(num_parts)]
    return parts

# Apply data augmentation to create augmented dataset
for audio_file, label in raw_audio_full.items():
    parts = split_audio(audio_file)
    for part in parts:
        raw_audio_augmented[part] = label


# In[9]:


import numpy as np

# Define a fixed length for MFCC feature vectors
max_length = 100

# Extract MFCC Features and Chop audio
# Preemphasis filter for high frequency
def preemphasis_filter(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

# Frame the signal into 25 ms frame and 10 ms frame shift
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
    
# Apply Hamming Windows and Compute the power spectrum
def power_spectrum(frames, NFFT=512):
    frames *= np.hamming(frames.shape[1])
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames

# Apply Mel Filterbank
def mel_filterbank(spectrum, num_filters=40, sampling_rate=22050, n_fft=512):
    mel_filterbank = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_filters)
    return np.dot(mel_filterbank, spectrum.T).T

# Compute MFCCS
def mfcc(signal, sampling_rate=22050, frame_length=512, frame_shift=256, num_mfcc=13, n_mels=40):
    emphasized_signal = preemphasis_filter(signal)
    framed_signal = frame_signal(emphasized_signal, frame_length, frame_shift)
    spectrum = power_spectrum(framed_signal)
    mel_spectrum = mel_filterbank(spectrum, num_filters=n_mels, sampling_rate=sampling_rate, n_fft=512)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrum), n_mfcc=num_mfcc, n_mels=n_mels)

    # Ensure the MFCCs have a fixed length
    if mfccs.shape[1] < max_length:
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs_padded = mfccs[:, :max_length]
    else:
        mfccs_padded = mfccs
    
    return mfccs_padded.flatten()


# In[14]:


import numpy as np
import librosa

# Feature Extraction: MFCC simplified
def extract_mfcc_augmented(audio_part, sigr=22050, n_mfcc=13, hop_length=int(0.010 * sr), n_fft=int(0.025 * sr)):
    audio_part = np.array(audio_part.get_array_of_samples(), dtype=np.float32)
    mfccs = librosa.feature.mfcc(y=audio_part, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfccs.T  # Transpose to have time steps as rows


# In[66]:


# Extract MFCC Features and create DataFrame
X_full = []
y_full = []

for audio_file, label in raw_audio_full.items():
    signal, sr = librosa.load(audio_file, sr=None)
    mfcc_features_full = mfcc(signal, sampling_rate=sr)
    X_full.append(mfcc_features.flatten())
    y_full.append(label)

import pandas as pd

# Convert the features and labels to a DataFrame
df_full_audio = pd.DataFrame(X_full)
df_full_audio = df_full_audio.fillna(0)
df_full_audio['label'] = y_full  # Add the labels as a new column

# Save the DataFrame to a CSV file
df_full_audio.to_csv('full_audio_features.csv', index=False)


# In[13]:


X_augmented = []
y_augmented = []

for audio_part, label in raw_audio_augmented.items():
    mfcc_features_augmented = extract_mfcc_augmented(audio_part)  # Replace with your simplified method function call
    X_augmented.append(mfcc_features_augmented.flatten())
    y_augmented.append(label)

df_augment_audio = pd.DataFrame(X_augmented)
df_augment_audio = df_augment_audio.fillna(0)
df_augment_audio['label'] = y_augmented  # Add the labels as a new column

# Save the DataFrame to a CSV file
df_full_audio.to_csv('augmented_audio_features.csv', index=False)


# In[77]:


# Combine full and augmented datasets
X_combined = X_full + X_augmented
y_combined = y_full + y_augmented

# Create DataFrame from features
df_combined = pd.DataFrame(X_combined)
df_combined = df_combined.fillna(0)
df_combined['label'] = y_combined

# Save to CSV
df_combined.to_csv('combined_audio_dataset.csv', index=False)


# In[78]:


# Load combined dataset
df_combined = pd.read_csv('combined_audio_dataset.csv')

# Preprocess Data
label_encoder = LabelEncoder()
df_combined['label'] = label_encoder.fit_transform(df_combined['label'])
X = df_combined.drop('label', axis=1).values
y = df_combined['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[79]:


# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Models
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC(kernel='rbf', C=12, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=32, criterion='entropy', random_state=42)
mlp_model = MLPClassifier(alpha=0.01, max_iter=1000, hidden_layer_sizes=(12,), solver='adam', random_state=42)


# In[81]:


# Train Models on Combined Data
models = [knn_model, svm_model, rf_model, mlp_model]
model_names = ['k-NN', 'SVM', 'Random Forest', 'MLP']
for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


# In[85]:


from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Define a pipeline with MinMaxScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32),
                          activation='relu',
                          solver='adam',
                          alpha=0.001,  # L2 penalty
                          batch_size=32,
                          learning_rate_init=0.001,
                          max_iter=500,
                          random_state=42))
])

# Train the model using the pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"MLP with Dropout & L2 Regularization - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

