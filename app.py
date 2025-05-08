import gradio as gr
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tempfile
import os

# Load the trained model, label encoder, and scaler
with open("heartbeat_xgboost_optimized.pkl", "rb") as f:
    model, label_encoder, scaler = pickle.load(f)

# Extract features from audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=20.0), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10, fmax=sr / 2), axis=1)
        return np.hstack([mfcc, chroma, contrast, [zcr, rmse], mel]), y
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None

# Plot audio waveform
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

# Predict heart condition
def predict(audio_file):
    if audio_file is None:
        return "No audio uploaded", None
    features, y = extract_features(audio_file)
    if features is None:
        return "Error extracting features", None
    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        waveform_path = plot_waveform(y)
        return predicted_label, waveform_path
    except Exception as e:
        return f"Prediction error: {e}", None

# Info block with teal-colored headers
info_block = """
<div style="
    border: 2px solid #00B3B3;
    border-radius: 15px;
    padding: 20px;
    background-color: #ffffff;
    color: #333333;
    font-size: 16px;
    max-width: 1200px;
    margin: 0 auto;
">

<h2 style="color:#00B3B3;">Why Choose Us?</h2>
<ul>
    <li>Easy to Use: No medical expertise required.</li>
    <li>Cost Effective: Pre-screen your heart at home for free.</li>
    <li>Smart Technology: Accurate ML-powered prediction from heartbeat data.</li>
</ul>

<h2 style="color:#00B3B3;">User Guide – How to Record Your Heart Sound</h2>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Heart_Auscultation_Locations.svg/1280px-Heart_Auscultation_Locations.svg.png" 
     alt="Body Position" 
     style="width: 100%; max-width: 500px; display: block; margin: 20px auto;">

<ul>
    <li>Place the chest piece at the apex part of your chest.</li>
    <li>Record your heartbeat for at least 9 seconds.</li>
    <li>Upload the audio file.</li>
    <li>Click <b>Submit</b> to see your result.</li>
</ul>

<h2 style="color:#00B3B3;">Understanding Your Results</h2>
<ul>
    <li>Murmur: Abnormal heart sounds due to turbulent blood flow. May indicate valve disorders.</li>
    <li>Normal:"Lub-dub" sounds indicating healthy heart valves.</li>
    <li>Extra Heart Sounds (Extrahls): May indicate heart failure, stiffness, or rapid filling of ventricles.</li>
    <li>Extrasystole: Extra heartbeat that interrupts regular rhythm. Often harmless but seek help if discomfort arises.</li>
</ul>

<h2 style="color:#00B3B3;">Contact Us</h2>
<div style="
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    background: #E0F7FA;
    padding: 15px;
    border-radius: 12px;
    justify-content: center;
    font-size: 15px;
">
    <div style="display:flex;align-items:center;gap:8px;">
        <img src="https://img.icons8.com/ios-filled/20/000000/new-post.png"/> khushipandey1133@gmail.com
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <img src="https://img.icons8.com/ios-filled/20/000000/new-post.png"/> varshabramhankar942@gmail.com
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <img src="https://img.icons8.com/ios-filled/20/000000/new-post.png"/> khushibothara2208@gmail.com
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <img src="https://img.icons8.com/ios-filled/20/000000/new-post.png"/> runishkayui@gmail.com
    </div>
</div>

<p style="margin-top:10px;">💬 Response Time: Within 24–48 hours</p>
</div>
"""

# Background watermark
css = """
body::before {
    content: "MEDICAL";
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-30deg);
    font-size: 300px;
    font-weight: 900;
    color: rgba(0, 0, 0, 0.04);
    z-index: 0;
    pointer-events: none;
    user-select: none;
    font-family: 'Segoe UI', sans-serif;
    white-space: nowrap;
}
"""

# Gradio Interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Render provides PORT env var
    gr.Interface(
        fn=predict,
        inputs=gr.Audio(label="🎧 Upload Heartbeat Audio", type="filepath"),
        outputs=[gr.Textbox(label="🩺 Predicted Heart Condition"), gr.Image(label="📊 Waveform")],
        title="<h1 style='color:#00B3B3;'>❤ Heartbeat Condition Detection</h1>",
        description="""<div style="position: relative; z-index: 1; font-size: 16px; color: #333;">
            <b>Predict Your Heart Health</b><br><br>
            Prevention is better than cure. Use our advanced AI to analyze your heart sounds and assess your cardiac health.<br><br>
            👉 <b>Click Submit after uploading a .wav file.</b><br><br>
        </div>""",
        article=info_block,
        css=css,
        theme="huggingface"
    ).launch(server_name="0.0.0.0", server_port=port)





