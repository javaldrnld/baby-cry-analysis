import os
import sys

import numpy as np

from src.data import augment_audio, make_dataset
from src.features import build_features
from src.models import predict_model, train_model
from src.utils import audio_processing
from src.visualization import visualize

# Load and prepare data
raw_audio = make_dataset.load_raw_audio(
    ["belly_pain", "burping", "discomfort", "hungry", "tired"],
    "../data/audio_augmented_raw/",
)
augmented_audio = augment_audio.augment_dataset(raw_audio)

# Initialize features and label list
X = []
y = []

# Extract features
for audio_part, label in augmented_audio.items():
    features = build_features.extract_features(audio_part)
    X.append(features)
    y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train models
(X_train, X_test, y_train, y_test), label_encoder = train_model.preprocess_data(X, y)
models, scaler = train_model.train_models(X_train, y_train)

# Evaluate models
for name, model in models.items():
    accuracy, precision, recall, f1, report = predict_model.evaluate_model(
        model, X_test, y_test, scaler
    )
    print(
        f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
    )
    print(f"{name} Classification Report:\n")
    print(report)

# Visualize results
y_pred = models["Random Forest"].predict(scaler.transform(X_test))
visualize.plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
