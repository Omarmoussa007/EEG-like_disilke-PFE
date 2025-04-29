import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
import os

# --------------------------- Fonctions Utilitaires ----------------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def extract_alpha_power(signal, fs):
    alpha_band = bandpass_filter(signal, 8, 12, fs)
    power = np.mean(alpha_band ** 2)
    return power

def extract_features_from_file(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    fs = 128  # fréquence d’échantillonnage en Hz
    f3 = data['F3'].values
    f4 = data['F4'].values

    # Découpe du signal en fenêtres de 1 seconde
    window_size = fs
    step = fs
    features = []

    for start in range(0, len(f3) - window_size + 1, step):
        f3_window = f3[start:start + window_size]
        f4_window = f4[start:start + window_size]
        f3_power = extract_alpha_power(f3_window, fs)
        f4_power = extract_alpha_power(f4_window, fs)
        features.append([f3_power, f4_power])

    # Aplatir en vecteur
    features = np.array(features).flatten()
    return features

# --------------------------- Interface Streamlit ----------------------------

st.set_page_config(page_title="EEG Classifier", layout="wide")

st.title("EEG Like/Dislike Classifier")

st.sidebar.header("Paramètres")
theme_mode = st.sidebar.radio("Mode d'affichage", ["Sombre", "Clair"])
uploaded_file = st.sidebar.file_uploader("Choisir un fichier EEG (.txt)", type="txt")

if uploaded_file is not None:
    st.subheader("Signal EEG")
    data = pd.read_csv(uploaded_file, delimiter="\t")
    
    st.write("Aperçu des données :")
    st.dataframe(data.head())

    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    ax[0].plot(data["F3"], color='blue')
    ax[0].set_title("F3 - Alpha")
    ax[1].plot(data["F4"], color='green')
    ax[1].set_title("F4 - Alpha")
    st.pyplot(fig)

    st.subheader("Classification")

    # Chargement du modèle
    try:
        model = joblib.load("modele_random_forest.pkl")
    except Exception as e:
        st.error("Erreur de chargement du modèle : " + str(e))
        st.stop()

    try:
        features = extract_features_from_file(uploaded_file)
        if len(features) != model.n_features_in_:
            st.error(f"Le modèle attend {model.n_features_in_} caractéristiques, mais {len(features)} ont été extraites.")
            st.stop()
        prediction = model.predict([features])[0]
        st.success(f"Résultat de la classification : **{prediction.upper()}**")
    except Exception as e:
        st.error("Erreur lors de la classification : " + str(e))

else:
    st.info("Veuillez charger un fichier EEG pour commencer.")
