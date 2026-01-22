import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = "RAVDESS Emotional speech audio"
MODEL_PATH = "models/audio_model.h5"
DURATION = 3
N_MFCC = 40

def load_data(data_path):
    features = []
    labels = []
    
    # 02 = Calm -> 1 (Confident)
    # 06 = Fearful -> 0 (Nervous)
    
    print("Scanning for files...")
    for root, _, files in os.walk(data_path):
        for file in files:
            if not file.endswith(".wav"):
                continue
            
            # Labeling Logic
            label = None
            if "-02-" in file:
                label = 1 # Confident
            elif "-06-" in file:
                label = 0 # Nervous
            
            if label is not None:
                file_path = os.path.join(root, file)
                try:
                    # Load audio
                    y, sr = librosa.load(file_path, duration=DURATION, sr=None)
                    
                    # Pad if shorter than duration (optional, but good for robustness)
                    # For this specific logic, we are taking the mean, so length consistency strictly involves duration load
                    # Librosa load with duration truncates. If short, it loads what's there.
                    
                    # Extract MFCCs
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                    
                    # Take mean across time to get shape (40,)
                    mfccs_mean = np.mean(mfccs.T, axis=0)
                    
                    features.append(mfccs_mean)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    print(f"Found {len(features)} samples.")
    return np.array(features), np.array(labels)

def train_model():
    # Load Data
    X, y = load_data(DATA_PATH)
    
    if len(X) == 0:
        print("No data found. Please check the 'RAVDESS Emotional speech audio' folder structure.")
        return

    # Reshape for CNN Input: (Batch, Steps, Channels) -> (Batch, 40, 1)
    X = np.expand_dims(X, axis=-1)
    
    # One-hot encode labels
    y = to_categorical(y, num_classes=2)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build Model
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(N_MFCC, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train
    print("Starting training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
