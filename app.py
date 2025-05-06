import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import json
import hashlib
from scipy.signal import butter, filtfilt, welch

# === CONFIGURATION ===
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

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# === CONSTANTES ===
FS = 250
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'Cz', 'C3', 'C4', 'T7', 'T8', 'O1', 'O2', 'Pz', 'POz']
BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
MODEL_PATH = r'C:\Users\21628\Desktop\eeg_project\eeg_rf_model.joblib'

# === GESTION DES COMPTES ===
USER_DATA_FILE = r'C:\Users\21628\Desktop\eeg_project\users_data.json'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_user_data():
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f)

def register(username, password):
    users = load_user_data()
    if username in users:
        return False
    users[username] = {"password": hash_password(password)}
    save_user_data(users)
    return True

def login(username, password):
    users = load_user_data()
    return username in users and users[username]["password"] == hash_password(password)

# === TRAITEMENT EEG ===
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
    powers = {band: np.trapz(psd[(freqs >= low) & (freqs <= high)], freqs[(freqs >= low) & (freqs <= high)]) for band, (low, high) in BANDS.items()}
    return powers

def extract_features(eeg):
    features = []
    for ch in range(eeg.shape[1]):
        signal = bandpass(eeg[:, ch])
        features.extend([np.mean(signal), np.std(signal)] + list(compute_bandpowers(signal).values()))
    return features

def predict_eeg(features):
    model = joblib.load(MODEL_PATH)
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0], model.predict_proba(features)[0]

# === INTERFACE ===
st.title("🧠 EEG Like/Dislike Classifier")

# Connexion / inscription
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['user_stats'] = {"like": 0, "dislike": 0}

if not st.session_state['logged_in']:
    option = st.sidebar.selectbox("Action", ["Inscription", "Connexion"])
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if option == "Inscription":
        confirm_password = st.text_input("Confirmer le mot de passe", type="password")
        if st.button("S'inscrire"):
            if password == confirm_password:
                if register(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success("Inscription réussie. Connecté.")
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur déjà pris.")
            else:
                st.error("Les mots de passe ne correspondent pas.")
    else:
        if st.button("Se connecter"):
            if login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Connexion réussie.")
                st.rerun()
            else:
                st.error("Identifiants incorrects.")
else:
    st.sidebar.write(f"Bienvenue, {st.session_state['username']}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['user_stats'] = {"like": 0, "dislike": 0}
        st.rerun()

    menu = st.sidebar.radio("Navigation", ["Accueil", "Prédiction", "Statistiques"])

    if menu == "Accueil":
        st.markdown("""
        ## 🎯 Objectif
        Classer les émotions EEG en LIKE ou DISLIKE à l'aide d'un modèle Random Forest.
        """)

    elif menu == "Prédiction":
        uploaded_file = st.file_uploader("Importer un fichier EEG (.txt)", type=["txt"])
        if uploaded_file:
            eeg = pd.read_csv(uploaded_file, sep=' ', header=None)
            if eeg.shape[1] != 14:
                st.error("Le fichier doit contenir 14 canaux EEG.")
            else:
                eeg.columns = CHANNELS
                features = extract_features(eeg.values)
                prediction, confidence = predict_eeg(features)
                label = "LIKE" if prediction == 1 else "DISLIKE"

                # Statistiques personnelles
                if prediction == 1:
                    st.session_state['user_stats']['like'] += 1
                else:
                    st.session_state['user_stats']['dislike'] += 1

                st.success(f"Prédiction : {label}")
                st.metric("Confiance LIKE", f"{confidence[1]*100:.2f}%")
                st.metric("Confiance DISLIKE", f"{confidence[0]*100:.2f}%")

                # Affichage des signaux Alpha F3 et F4
                alpha_F3 = bandpass_alpha(eeg['F3'].values)
                alpha_F4 = bandpass_alpha(eeg['F4'].values)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=alpha_F3, mode='lines', name='F3 (Alpha)', line=dict(color='cyan')))
                fig.add_trace(go.Scatter(y=alpha_F4, mode='lines', name='F4 (Alpha)', line=dict(color='magenta')))
                fig.update_layout(title='Signaux Alpha - F3 et F4', xaxis_title='Échantillons', yaxis_title='Amplitude', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

    elif menu == "Statistiques":
        stats = st.session_state['user_stats']
        st.metric("Nombre de LIKE", stats['like'])
        st.metric("Nombre de DISLIKE", stats['dislike'])

        fig = px.bar(
            x=["LIKE", "DISLIKE"],
            y=[stats['like'], stats['dislike']],
            labels={'x': "Émotions", 'y': "Nombre de prédictions"},
            color=["LIKE", "DISLIKE"],
            color_discrete_map={"LIKE": "cyan", "DISLIKE": "magenta"},
            title="Distribution des Prédictions"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
