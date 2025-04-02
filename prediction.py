import os
import librosa
import numpy as np
import pickle

# 🎯 Load the trained model, label encoder, and scaler
with open("heartbeat_xgboost_optimized.pkl", "rb") as f:
    model, label_encoder, scaler = pickle.load(f)

# 🔍 Feature extraction function
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file

        # Compute features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=20.0), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10, fmax=sr / 2), axis=1)

        return np.hstack([mfcc, chroma, contrast, [zcr, rmse], mel])
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# 📌 Predict heartbeat condition for all audio files in a folder
def predict_heartbeat_folder(folder_path):
    if not os.path.isdir(folder_path):
        print("⚠️ Provided path is not a valid directory.")
        return

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):  # Process only .wav files
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is None:
                continue

            # Normalize using the same scaler as training
            features_scaled = scaler.transform([features])

            # Predict label
            prediction = model.predict(features_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            print(f"🔍 {file_name}: {predicted_label}")

# 📂 Example usage
if __name__ == "__main__":
    folder_path = r"D:\Projects\IT_Major_Project\new_mvp_sahil_mohurale\audio_dataset_filtered"
    predict_heartbeat_folder(folder_path)