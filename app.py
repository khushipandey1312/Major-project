# from flask import Flask, request, render_template, jsonify
# import librosa
# import numpy as np
# import pickle

# # 🎯 Load the trained model, label encoder, and scaler
# with open("new_mvp_sahil_mohurale/heartbeat_xgboost_optimized.pkl", "rb") as f:
#     model, label_encoder, scaler = pickle.load(f)

# # Initialize Flask app
# app = Flask(__name__)

# # 🔍 Feature extraction function
# def extract_features(file_path):
#     try:
#         y, sr = librosa.load(file_path, sr=None)  # Load audio file

#         # Compute features with correct fmin and fmax
#         mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
#         chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
#         contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=20.0), axis=1)
#         zcr = np.mean(librosa.feature.zero_crossing_rate(y))
#         rmse = np.mean(librosa.feature.rms(y=y))
#         mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10, fmax=sr / 2), axis=1)

#         return np.hstack([mfcc, chroma, contrast, [zcr, rmse], mel])
#     except Exception as e:
#         print(f"Error extracting features: {e}")
#         return None

# # 📌 Predict heartbeat condition
# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     error = None

#     if request.method == "POST":
#         if "file" not in request.files:
#             error = "No file part"
#             if request.headers.get("X-Requested-With") == "XMLHttpRequest":
#                 return jsonify(error=error)
#             return render_template("index.html", prediction=prediction, error=error)

#         audio_file = request.files["file"]
#         if audio_file.filename == "":
#             error = "No selected file"
#             if request.headers.get("X-Requested-With") == "XMLHttpRequest":
#                 return jsonify(error=error)
#             return render_template("index.html", prediction=prediction, error=error)

#         # Save the uploaded file temporarily
#         file_path = "temp_audio.wav"
#         audio_file.save(file_path)

#         features = extract_features(file_path)
#         if features is None:
#             error = "Error extracting features. Check the audio file."
#             if request.headers.get("X-Requested-With") == "XMLHttpRequest":
#                 return jsonify(error=error)
#             return render_template("index.html", prediction=prediction, error=error)

#         # Normalize using the same scaler as training
#         features_scaled = scaler.transform([features])

#         # Predict label
#         prediction = model.predict(features_scaled)
#         predicted_label = label_encoder.inverse_transform(prediction)[0]

#         # Return JSON response for AJAX requests
#         if request.headers.get("X-Requested-With") == "XMLHttpRequest":
#             return jsonify(prediction=predicted_label, error=error)
        
#         # Render template for non-AJAX requests
#         return render_template("index.html", prediction=predicted_label, error=error)

#     return render_template("index.html", prediction=prediction, error=error)

# # Start the Flask app
# if __name__ == "__main__":
#     app.run(debug=True)



import base64
import io
from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
import pickle
import wave

# 🎯 Load the trained model, label encoder, and scaler
with open("new_mvp_sahil_mohurale/heartbeat_xgboost_optimized.pkl", "rb") as f:
    model, label_encoder, scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

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
        print(f"Error extracting features: {e}")
        return None

# 📌 Predict heartbeat condition
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        # Handle uploaded file or recorded audio (all passed as 'file')
        if "file" in request.files and request.files["file"].filename != "":
            audio_file = request.files["file"]
            file_path = "temp_audio.wav"
            audio_file.save(file_path)
        else:
            error = "No audio uploaded or recorded."
            return jsonify(error=error)

        # Extract features
        features = extract_features(file_path)
        if features is None:
            error = "Error extracting features. Check the audio file."
            return jsonify(error=error)

        # Normalize and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Return JSON response
        return jsonify(prediction=predicted_label, error=error)

    return render_template("index.html", prediction=prediction, error=error)

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
