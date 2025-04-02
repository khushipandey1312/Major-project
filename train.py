

import os
import librosa
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# 📂 Directory containing audio files
AUDIO_DIR = "audio_dataset_filtered"

# 🎯 Extract label from filename
def extract_label(filename):
    parts = filename.split('_')
    for i, part in enumerate(parts):
        if part.isdigit():
            return '_'.join(parts[:i])  # Label is everything before the first number
    return None

# 🔍 Extract features with **fixed spectral parameters**
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio

        # Compute features safely
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=20.0), axis=1)  # Fixed fmin
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10, fmax=sr / 2), axis=1)  # Safe fmax

        return np.hstack([mfcc, chroma, contrast, [zcr, rmse], mel])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 📈 Prepare dataset
X, y = [], []
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        label = extract_label(filename)
        if label:
            features = extract_features(os.path.join(AUDIO_DIR, filename))
            if features is not None:
                X.append(features)
                y.append(label)

if len(X) == 0:
    raise ValueError("No valid features extracted! Check dataset.")

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

# 🔥 XGBoost with Randomized Search
xgb_model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9, 12],
    'subsample': [0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0, 0.01, 0.1, 1, 10]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,  # Number of random combinations to try
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 🏋️ Train the optimized model
random_search.fit(X_train, y_train)

# Best model from tuning
best_model = random_search.best_estimator_

# 🧪 Evaluate performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Optimized Model Accuracy: {accuracy:.2f}")

# 💾 Save the trained model
with open("heartbeat_xgboost_optimized.pkl", "wb") as f:
    pickle.dump((best_model, label_encoder, scaler), f)

print("🎉 Optimized model saved as heartbeat_xgboost_optimized.pkl")