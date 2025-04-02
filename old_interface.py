import gradio as gr
import numpy as np
import librosa
import joblib
import wave
import matplotlib.pyplot as plt
import tempfile


def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        y = librosa.util.normalize(y)
        if len(y) < 1024:
            y = np.pad(y, (0, max(0, 1024 - len(y))), mode='constant')
        n_fft = min(1024, len(y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        chroma_mean = np.mean(chroma.T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        try:
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_mean = np.mean(tonnetz.T, axis=0)
        except ValueError:
            tonnetz_mean = np.zeros(6)
        features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean))
        return features, y
    except Exception as e:
        return None, str(e)


def plot_waveform(y):
    plt.figure(figsize=(4, 2))
    plt.plot(y)
    plt.title("Heartbeat Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name


def calculate_bmi(weight, height):
    height_m = height * 0.3048  # Convert feet to meters
    return round(weight / (height_m ** 2), 2)


def predict_heart_condition(name, age, weight, gender, height, audio_file):
    model = joblib.load('heart_disease_voting_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    features, y = extract_features(audio_file)
    if features is None:
        return "Error extracting features", None, None, None

    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    bmi = calculate_bmi(weight, height)
    waveform_img = plot_waveform(y)

    report = f"""
    **Patient Report**
    Name: {name}
    Age: {age} years
    Gender: {gender}
    Weight: {weight} kg
    Height: {height} ft
    BMI: {bmi}
    Predicted Condition: {predicted_label}
    """
    report_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    with open(report_file.name, 'w') as f:
        f.write(report)

    return predicted_label, waveform_img, bmi, report_file.name


gr.Interface(
    fn=predict_heart_condition,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Number(label="Age (years)"),
        gr.Number(label="Weight (kg)"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Height (ft)"),
        gr.Audio(label="Upload Heartbeat Audio", type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Predicted Heart Condition"),
        gr.Image(label="Heartbeat Waveform"),
        gr.Number(label="BMI"),
        gr.File(label="Download Report")
    ],
    title="Heart Disease Prediction",
    description="Upload a heartbeat audio file along with personal details to predict heart condition."
).launch()
