#!/usr/bin/env python
# coding: utf-8

# IEEE Citation: K. Alam and K. A. Mamun, "From Cries to Answers: A Comprehensive CNN+DNN Hybrid Model for Infant Cry Classification with Enhanced Data Augmentation and Feature Extraction," 2024 International Conference on Advances in Computing, Communication, Electrical, and Smart Systems (iCACCESS), Dhaka, Bangladesh, 2024, pp. 1-6, doi: 10.1109/iCACCESS61735.2024.10499449.

# In[ ]:





# 

# In[1]:


import os
import subprocess

def convert_file(file_path):


  # Get the file extension.
  extension = os.path.splitext(file_path)[1]

  # Check if the file is an audio file.
  if extension in [".mp3", ".amr", ".m4a", ".mpa", ".aac",".ogg", ".3gpp"]:
    # Convert the file to WAV format.
    subprocess.call([
        "ffmpeg", "-i", file_path, "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2", "-b:a", "192k", os.path.join(os.path.dirname(file_path), os.path.basename(file_path) + ".wav")
    ])

def convert_folder(folder_path):


  # Get all files in the folder.
  files = os.listdir(folder_path)

  # Convert each file to WAV format.
  for file in files:
    convert_file(os.path.join(folder_path, file))

if __name__ == "__main__":
  # Get the folder path from the user.
  folder_path = "/home/bokuto/Documents/baby-cry-analysis/data/audio_augmented_raw"

  # Convert all files in the folder to WAV format.
  convert_folder(folder_path)


# In[2]:


import librosa
import os
import pandas as pd

data_dir = '/home/bokuto/Documents/baby-cry-analysis/data/audio_augmented_raw'
accent_folders = os.listdir(data_dir)
print(os.listdir(data_dir))




# In[3]:


data = []

for accent_folder in accent_folders:
    accent_path = os.path.join(data_dir, accent_folder)

    for file in os.listdir(accent_path):
        if file.endswith('.wav'):
            file_path = os.path.join(accent_path, file)
            audio, sr = librosa.load(file_path, sr=None)
            data.append((file_path, accent_folder))

df = pd.DataFrame(data, columns=['Path', 'Class'])


# In[4]:


df.head(5)


# In[6]:


# changing integers to actual region.
df['Class'] = df['Class'].replace({1:'belly_pain', 2:'burping', 3:'discomfort', 4:'hungry', 5:'tired'})

df.head()


# In[7]:


df.Class


# In[8]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:





# In[9]:


audio_data= df.Path
labels = df.Class


# In[10]:


audio_data = np.array(audio_data)
labels = np.array(labels)


# In[ ]:


get_ipython().system('pip install imbalanced-learn')


# In[11]:


import librosa
import numpy as np
from imblearn.over_sampling import SMOTE



# In[12]:


audio_data.shape


# In[13]:


import matplotlib.pyplot as plt
plt.title('Count of Class Data', size=16)
class_counts = df['Class'].value_counts()

# Plot the class counts using a bar plot
class_counts.plot(kind='bar')
plt.title('Class Counts')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[ ]:





# In[14]:


import matplotlib.pyplot as plt

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} Class'.format(e), size=15)
    plt.plot(data)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} Class'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()


# In[15]:


import librosa
from IPython.display import Audio
classes='hungry'
path = np.array(df.Path[df.Class==classes])[0]
data, sampling_rate = librosa.load(path)
create_spectrogram(data, sampling_rate, classes)
create_waveplot(data, sampling_rate, classes)
Audio(path)


# In[16]:


import librosa
from IPython.display import Audio
classes='belly_pain'
path = np.array(df.Path[df.Class==classes])[0]
data, sampling_rate = librosa.load(path)
create_spectrogram(data, sampling_rate, classes)
create_waveplot(data, sampling_rate, classes)
Audio(path)




# In[17]:


import librosa
from IPython.display import Audio
classes='burping'
path = np.array(df.Path[df.Class==classes])[0]
data, sampling_rate = librosa.load(path)
create_spectrogram(data, sampling_rate, classes)
create_waveplot(data, sampling_rate, classes)
Audio(path)


# In[18]:


import librosa
from IPython.display import Audio
classes='discomfort'
path = np.array(df.Path[df.Class==classes])[0]
data, sampling_rate = librosa.load(path)
create_spectrogram(data, sampling_rate, classes)
create_waveplot(data, sampling_rate, classes)
Audio(path)


# 'laugh', 6:'noise', 7:'silence', 8:'burping'

# In[19]:


import librosa
from IPython.display import Audio
classes='tired'
path = np.array(df.Path[df.Class==classes])[0]
data, sampling_rate = librosa.load(path)
create_spectrogram(data, sampling_rate, classes)
create_waveplot(data, sampling_rate, classes)
Audio(path)


# In[20]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics


# **Data Augmentation**

# In[24]:


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate):
    return librosa.effects.pitch_shift(data, sampling_rate, rate=0.7)

def time_masking(data, max_mask_time=100):
    mask_time = np.random.randint(0, max_mask_time)
    data[:mask_time] = 0
    return data

def frequency_masking(data, max_mask_freq=50):
    mask_freq = np.random.randint(0, max_mask_freq)
    data_freq = np.fft.fft(data)
    data_freq[:mask_freq] = 0
    return np.real(np.fft.ifft(data_freq))

# taking any example and checking for techniques.
path = np.array(df.Path)[1]
data, sample_rate = librosa.load(path)

def augment(data, sampling_rate):
    augmentations = [noise, stretch, shift, pitch, time_masking, frequency_masking]
    aug_choice = np.random.choice(augmentations)
    
    if aug_choice == pitch:
        return aug_choice(data, sampling_rate)
    else:
        return aug_choice(data)
augment(data, sample_rate)


# In[25]:


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    #data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(new_data)
    result = np.vstack((result, res3)) # stacking vertically

    return result


# In[26]:


X, Y = [], []
c=0
for path, Class in zip(df.Path, df.Class):
    feature = get_features(path)
    for ele in feature:
        c = c+1
        print(c)
        X.append(ele)
        # appending class 3 times as we have made 7 augmentation techniques on each audio file.
        Y.append(Class)


# In[27]:


len(X), len(Y), df.Path.shape


# In[28]:


Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)
Features.head()


# In[29]:


X = Features.iloc[: ,:-1].values
Y = Features['labels'].values


# In[30]:


# Apply SMOTE to balance classes

smote = SMOTE(random_state=42)
X_smote, Y_smote = smote.fit_resample(X, Y)


# In[31]:


# Create a new DataFrame with the balanced data
balanced_data = pd.DataFrame(X_smote)
balanced_data['labels'] = Y_smote

# Save the balanced data to a new CSV file
balanced_data.to_csv('balanced_features.csv', index=False)


# In[32]:


import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the features and labels from the balanced CSV file
balanced_data = pd.read_csv('balanced_features.csv')

# Separate features and labels
X_balanced = balanced_data.drop('labels', axis=1)
Y_balanced = balanced_data['labels']

# Count the occurrences of each class before and after SMOTE
balanced_class_counts = Y_balanced.value_counts().sort_index()

# Plotting the class distributions
plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 2)
plt.bar(balanced_class_counts.index, balanced_class_counts.values)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[33]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# In[34]:


# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y_balanced = encoder.fit_transform(np.array(Y_balanced).reshape(-1,1)).toarray()


# In[35]:


# splitting data
x_train, x_test, y_train, y_test = train_test_split(X_balanced, Y_balanced, random_state=0,test_size = 0.20, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[36]:


# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[37]:


# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[38]:


import tensorflow.keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint


# In[39]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score



# In[40]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

model = Sequential()

# CNN layers
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))


model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=4))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=6))

# Flatten before passing to Dense layers
model.add(Flatten())

# DNN layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
# Final output layer
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[41]:


rlrp = ReduceLROnPlateau(monitor='loss', factor=0.5, verbose=0, patience=2, min_lr=0.0001)


# Cross Validation

# In[42]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Initialize lists to store aggregated predictions and actual labels
all_predicted_labels = []
all_actual_labels = []
# Create lists to store results from each fold
accuracies = []
losses = []
# Define the number of folds
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds)

# Iterate through each fold
for fold, (train_index, val_index) in enumerate(skf.split(X_balanced, Y_balanced.argmax(axis=1))):
    print(f"Fold {fold + 1}/{k_folds}")

    # Split data into train and validation sets for this fold
    x_train, x_val = X_balanced.iloc[train_index], X_balanced.iloc[val_index]
    y_train, y_val = Y_balanced[train_index], Y_balanced[val_index]

    # Perform any necessary data preprocessing (scaling, reshaping)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    x_train_scaled = np.expand_dims(x_train_scaled, axis=2)
    x_val_scaled = np.expand_dims(x_val_scaled, axis=2)

 # Build your model
    model = Sequential()

    # CNN layers
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(x_train_scaled.shape[1], x_train_scaled.shape[2])))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=6))


    # Flatten before passing to Dense layers
    model.add(Flatten())

    # DNN layers
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))

    # Final output layer
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    history = model.fit(x_train_scaled, y_train, epochs=30, batch_size=64, validation_data=(x_val_scaled, y_val), callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, verbose=0, patience=2, min_lr=0.0001)])

    # Evaluate the model on validation set
    _, accuracy = model.evaluate(x_val_scaled, y_val)
    accuracies.append(accuracy)
    losses.append(history.history['loss'][-1])

    # Make predictions on the validation set for this fold
    fold_predictions = model.predict(x_val_scaled)

    # Convert one-hot encoded labels back to categorical labels
    fold_predicted_labels = np.argmax(fold_predictions, axis=1)
    fold_actual_labels = np.argmax(y_val, axis=1)

    # Append fold predictions and actual labels to the aggregated lists
    all_predicted_labels.extend(fold_predicted_labels)
    all_actual_labels.extend(fold_actual_labels)

# Generate a classification report and confusion matrix from aggregated predictions
print("\nClassification Report:")
print(classification_report(all_actual_labels, all_predicted_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(all_actual_labels, all_predicted_labels))

# Print average accuracy and loss across all folds
print(f"Average accuracy: {sum(accuracies) / len(accuracies)}")
print(f"Average loss: {sum(losses) / len(losses)}")
model.summary()


# In[ ]:


import matplotlib.pyplot as plt

# Plotting accuracy and loss trends across folds
plt.figure(figsize=(12, 5))

# Plotting accuracies
plt.subplot(1, 2, 1)
plt.plot(range(1, k_folds + 1), accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy across Folds')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, k_folds + 1))
plt.grid()

# Plotting losses
plt.subplot(1, 2, 2)
plt.plot(range(1, k_folds + 1), losses, marker='o', linestyle='-', color='r')
plt.title('Loss across Folds')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.xticks(np.arange(1, k_folds + 1))
plt.grid()

plt.tight_layout()
plt.show()


# In[ ]:


# predicting on test data.
pred_test = model.predict(x_val_scaled)
y_pred = encoder.inverse_transform(pred_test)

y_test1 = encoder.inverse_transform(y_val)


# In[ ]:


df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = all_predicted_labels
df['Actual Labels'] = all_actual_labels

df


# In[ ]:


import seaborn as sns
cm = confusion_matrix(all_actual_labels, all_predicted_labels)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()


# In[ ]:


print(classification_report(all_actual_labels, all_predicted_labels))


# In[44]:


jupyter nbconvert --to script 'my-notebook.ipynb'

