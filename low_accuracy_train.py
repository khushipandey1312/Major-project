import os
import librosa
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Directory containing audio files
AUDIO_DIR = "audio_dataset"


# Function to extract label from filename
def extract_label(filename):
    """Extracts label from filename before the first occurrence of '_number'."""
    parts = filename.split('_')

    for i, part in enumerate(parts):
        if part.isdigit():  # First numeric part found, label is before this
            return '_'.join(parts[:i])

    return None  # If no numeric part found


# Function to extract multiple audio features
def extract_features(file_path):
    """Extracts multiple features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio with original sampling rate

        # Dynamic adjustment of spectral contrast parameters
        fmin = max(20, sr * 0.01)  # Ensure fmin is at least 20 Hz
        n_bands = min(5, sr // (2 * fmin))  # Keep n_bands within Nyquist limit

        # MFCC Features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Spectral Contrast (Now dynamically adjusted)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, fmin=fmin)
        contrast_mean = np.mean(contrast, axis=1)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # Root Mean Square Energy (RMSE)
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        mel_mean = np.mean(mel, axis=1)

        # Combine all features
        features = np.hstack([mfcc_mean, chroma_mean, contrast_mean, [zcr_mean, rmse_mean], mel_mean])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Prepare dataset
X, y = [], []
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):  # Ensure it's an audio file
        label = extract_label(filename)
        if label:  # Ignore files where label extraction fails
            features = extract_features(os.path.join(AUDIO_DIR, filename))
            if features is not None and len(features) > 0:  # Ensure valid features
                X.append(features)
                y.append(label)

# Check if we have valid data
if len(X) == 0:
    raise ValueError("No valid features extracted. Check your dataset and preprocessing steps.")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("heartbeat_xgboost_model.pkl", "wb") as f:
    pickle.dump((model, label_encoder, scaler), f)

print("Model saved as heartbeat_xgboost_model.pkl")
