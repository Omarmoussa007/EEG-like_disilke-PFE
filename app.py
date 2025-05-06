import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import base64
import streamlit.components.v1 as components
from scipy.signal import butter, filtfilt, welch

# === CONFIG ===
st.set_page_config(page_title="EEG Like/Dislike Classifier", page_icon="🧠", layout="wide")

# === STYLE FUTURISTE ===
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #00fff7;
}

h1, h2, h3, h4 {
    text-align: center;
    color: #00fff7;
    text-shadow: 0 0 5px #00fff7, 0 0 10px #00fff7, 0 0 20px #00fff7;
    font-family: 'Orbitron', sans-serif;
}

button {
    background-color: #0e1117 !important;
    color: #00fff7 !important;
    border: 1px solid #00fff7 !important;
    border-radius: 10px !important;
    padding: 0.75em 1.5em !important;
}

button:hover {
    background-color: #00fff7 !important;
    color: #0e1117 !important;
    transition: 0.5s;
}

.sidebar .sidebar-content {
    background-color: #0e1117;
    color: #00fff7;
}

.css-1d391kg {
    background-color: #0e1117;
}

.st-bb {
    background-color: #0e1117;
}

.st-bc {
    background-color: #0e1117;
}

a {
    color: #00fff7;
}

footer {
    visibility: hidden;
}
/* Modification pour le mode clair */
@media (prefers-color-scheme: light) {
    button {
        color: black !important;
        background-color: #e0f7fa !important;
        border-color: #00bcd4 !important;
    }

    button:hover {
        background-color: #00bcd4 !important;
        color: white !important;
    }
}
</style>
""", unsafe_allow_html=True)


# === CONSTANTS ===
FS = 250
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'Cz', 'C3', 'C4', 'T7', 'T8', 'O1', 'O2', 'Pz', 'POz']
BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
MODEL_PATH = r'C:\Users\21628\Desktop\eeg_project\eeg_rf_model.joblib'
RESULTS_FILE = "results.csv"

# === FUNCTIONS ===
def bandpass(data, low=1, high=50, fs=FS, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def bandpass_alpha(signal, fs=FS, order=5):
    nyq = 0.5 * fs
    low, high = 8, 12
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def compute_bandpowers(signal, fs=FS):
    freqs, psd = welch(signal, fs=fs, nperseg=fs)
    powers = {}
    for band, (low, high) in BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        powers[band] = np.trapz(psd[idx], freqs[idx])
    return powers

def extract_features(eeg):
    features = []
    bandpower_list = []
    for ch in range(eeg.shape[1]):
        signal = eeg[:, ch]
        signal = bandpass(signal)
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        bandpowers = compute_bandpowers(signal)
        features.extend([mean_val, std_val] + list(bandpowers.values()))
        bandpower_list.append(bandpowers)
    return features, bandpower_list

def predict_eeg(features):
    model = joblib.load(MODEL_PATH)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]
    return prediction, confidence

# def play_sound():
#     sound_file = "sounds/blop.mp3"
#     if os.path.exists(sound_file):
#         with open(sound_file, "rb") as audio_file:
#             audio_bytes = audio_file.read()
#             b64 = base64.b64encode(audio_bytes).decode()
#             md = f"""
#             <audio autoplay>
#             <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#             </audio>
#             """
#             st.markdown(md, unsafe_allow_html=True)

# === INTERFACE ===
st.title("🧠 EEG Like/Dislike Classifier")
st.markdown("### Découvrez vos émotions grâce à votre cerveau... 🌟")

menu = st.sidebar.selectbox("Navigation", ["Accueil", "Prédiction", "Statistiques"])

if menu == "Accueil":
    st.header("Présentation du Projet 🎯")
    st.markdown("""
Ce projet vise à **classer les réactions émotionnelles** (**LIKE** ou **DISLIKE**) à partir de **signaux EEG** enregistrés chez différents utilisateurs.

Grâce aux techniques de **filtrage du signal**, **extraction de caractéristiques EEG** et **modèles de machine learning (Random Forest)**, nous réalisons une prédiction automatique des préférences cérébrales.

### 🛠️ Technologies utilisées :
- Python
- Streamlit
- Scikit-Learn
- SciPy
- Plotly

### 📈 Pipeline de traitement :
1. Importation fichier EEG (.txt)
2. Filtrage (1–50 Hz)
3. Extraction bandpowers
4. Modélisation Random Forest
5. Prédiction LIKE/DISLIKE

### 👨‍💻 Réalisé par :
- Ahmed Mlika — ahmedmlika@gmail.com
- Omar Moussa — omarmoussa967@gmail.com

Université : **ISSATS Sousse**

🌟 Merci pour votre attention !
""")
elif menu == "Prédiction":
    uploaded_file = st.file_uploader("Importer un fichier EEG (.txt)", type=["txt"])
    if uploaded_file is not None:
        with st.spinner('Analyse en cours... 🧠'):
            eeg = pd.read_csv(uploaded_file, sep=' ', header=None, engine='python')
            if eeg.shape[1] != 14:
                st.error("Le fichier doit avoir 14 canaux EEG.")
            else:
                eeg.columns = CHANNELS
                features, bandpowers_all = extract_features(eeg.values)
                prediction, confidence = predict_eeg(features)
                label = "LIKE" if prediction == 1 else "DISLIKE"
                confidence_like = round(confidence[1]*100, 2)
                confidence_dislike = round(confidence[0]*100, 2)

                st.success(f"Résultat : {label}")
                st.metric(label="Confiance LIKE", value=f"{confidence_like}%")
                st.metric(label="Confiance DISLIKE", value=f"{confidence_dislike}%")
                

                # === GRAPHIQUES ===
                st.subheader("Signal EEG Brut (toutes les électrodes)")
                fig_raw = go.Figure()
                for ch in CHANNELS:
                    fig_raw.add_trace(go.Scatter(y=eeg[ch], mode='lines', name=ch))
                fig_raw.update_layout(
                    xaxis_title="Temps (ms)", 
                    yaxis_title="Amplitude (μV)", 
                    height=400
                )
                st.plotly_chart(fig_raw, use_container_width=True)

                eeg_filtered = bandpass(eeg.values)
                st.subheader("Signal EEG Filtré (1-50 Hz)")
                fig_filt = go.Figure()
                for idx, ch in enumerate(CHANNELS):
                    fig_filt.add_trace(go.Scatter(y=eeg_filtered[:, idx], mode='lines', name=ch))
                fig_filt.update_layout(
                    xaxis_title="Temps (ms)", 
                    yaxis_title="Amplitude (μV)", 
                    height=400
                )
                st.plotly_chart(fig_filt, use_container_width=True)

                f3_raw = eeg['F3']
                f4_raw = eeg['F4']
                f3_filt = bandpass_alpha(f3_raw)
                f4_filt = bandpass_alpha(f4_raw)

                def compute_fft(signal, fs=FS):
                    n = len(signal)
                    freqs = np.fft.fftfreq(n, d=1/fs)
                    fft_values = np.fft.fft(signal)
                    idx = np.where(freqs >= 0)
                    return freqs[idx], np.abs(fft_values[idx])

                freqs_f3, fft_f3 = compute_fft(f3_filt)
                freqs_f4, fft_f4 = compute_fft(f4_filt)

                st.subheader("Canal F3 Brut")
                fig_f3_raw = px.line(y=f3_raw)
                fig_f3_raw.update_layout(
                    xaxis_title="Temps (ms)", 
                    yaxis_title="Amplitude (μV)", 
                    height=400
                )
                st.plotly_chart(fig_f3_raw, use_container_width=True)

                st.subheader("Canal F3 Filtré")
                fig_f3_filt = px.line(y=f3_filt)
                
                fig_f3_filt.update_layout(
                    xaxis_title="Temps (ms)", 
                    yaxis_title="Amplitude (μV)", 
                    height=400
                )
                st.plotly_chart(fig_f3_filt, use_container_width=True)

                st.subheader("Canal F4 Brut")
                fig_f4_raw = px.line(y=f4_raw)
                
                fig_f4_raw.update_layout(
                    xaxis_title="Temps (ms)", 
                    yaxis_title="Amplitude (μV)", 
                    height=400
                )
                st.plotly_chart(fig_f4_raw, use_container_width=True)
                
                st.subheader("Canal F4 Filtré")
                fig_f4_filt = px.line(y=f4_filt)
                
                fig_f4_filt.update_layout(
                    xaxis_title="Temps (ms)", 
                    yaxis_title="Amplitude (μV)", 
                    height=400
                )
                st.plotly_chart(fig_f4_filt, use_container_width=True)

                st.subheader("Spectre de Fréquence - Alpha F3")
                fig_fft_f3 = px.line(x=freqs_f3, y=fft_f3)
                fig_fft_f3.update_layout(title="FFT Alpha F3 (8-12 Hz)")
                fig_fft_f3.update_xaxes(range=[7,13])
                
                fig_fft_f3.update_layout(
                    xaxis_title="Frequence(Hz)", 
                    yaxis_title="Puissance(μV²)", 
                    height=400
                )
                st.plotly_chart(fig_fft_f3, use_container_width=True)

                st.subheader("Spectre de Fréquence - Alpha F4")
                fig_fft_f4 = px.line(x=freqs_f4, y=fft_f4)
                fig_fft_f4.update_layout(title="FFT Alpha F4 (8-12 Hz)")
                fig_fft_f4.update_xaxes(range=[7,13])
                
                fig_fft_f4.update_layout(
                    xaxis_title="Frequence(Hz)", 
                    yaxis_title="Puissance(μV²)", 
                    height=400
                )
                st.plotly_chart(fig_fft_f4, use_container_width=True)
                
                dt = 1 / FS
                power_f3 = np.trapz(f3_filt ** 2, dx=dt)
                power_f4 = np.trapz(f4_filt ** 2, dx=dt)

                st.subheader("Comparaison Puissance Alpha F3 vs F4")
                fig_compare = px.bar(
                    x=["Alpha F3", "Alpha F4"],
                    y=[power_f3, power_f4],
                    color=["Alpha F3", "Alpha F4"],
                    title="Taux de puissance Alpha"
                )
                fig_compare.update_layout(
                    xaxis_title="Alpha F3,Alpha F4", 
                    yaxis_title="Puissance(μV²)", 
                    height=400
                )
                st.plotly_chart(fig_compare, use_container_width=True)

                st.subheader("Résumé des Résultats")
                results_summary = pd.DataFrame({
                    "Puissance Alpha F3 (uV²)": [power_f3],
                    "Puissance Alpha F4 (uV²)": [power_f4],
                    "Résultat Prédiction": [label]
                })
                st.table(results_summary)

elif menu == "Statistiques":
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        st.subheader("Répartition LIKE/DISLIKE")
        fig1 = px.pie(df, names='prediction')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Distribution de confiance LIKE")
        fig2 = px.histogram(df, x='confidence_like', nbins=20)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df)
        st.download_button(
            label="Télécharger Résultats CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='results.csv',
            mime='text/csv'
        )
    else:
        st.info("Aucun résultat disponible pour l'instant.")

# === FOOTER FINAL ===
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "Réalisé avec ❤️ par <b>Ahmed Mlika</b> & <b>Omar Moussa</b><br>Université ISSATS Sousse"
    "</div>",
    unsafe_allow_html=True
)

