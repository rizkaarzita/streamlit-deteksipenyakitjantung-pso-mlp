import streamlit as st
import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import zipfile
import os

# Ekstrak dulu jika file zip
if not os.path.exists("model_mlp_pso_savedmodel"):
    with zipfile.ZipFile("model_mlp_pso_savedmodel.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Load dari SavedModel folder
model = tf.keras.models.load_model("model_mlp_pso_savedmodel")


# Load model dan scaler
# model = load_model("model_mlp_pso.keras")
scaler = joblib.load("scaler.save")

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded")

# CSS untuk styling
st.markdown("""
    <style>
    .main { background-color: #FFF5F5; }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .title { font-size: 36px; color: #FF4B4B; text-align: center; margin-bottom: 20px; }
    .header { font-size: 24px; color: #090a09; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title("🔍 Deteksi Dini Penyakit Jantung")


# Sidebar: Informasi tambahan di sidebar
st.sidebar.markdown("""
<div class="info-box">
    <h3>Informasi Penyakit Jantung</h3>
    <p>Penyakit jantung adalah salah satu penyebab kematian utama di dunia. Berikut variabel penelitian:</p>
    <ul>
        <li>age: usia pasien</li>
        <li>sex: jenis kelamin</li>
        <li>cp: jenis nyeri dada (0: tidak nyeri, 1: ringan, 2: sedang, 3: parah)</li>
        <li>trestbps: Tekanan darah saat istirahat (mm Hg)</li>
        <li>chol: Kadar kolesterol (mg/dl) (>200 berisiko)</li>
        <li>fbs: Gula darah puasa (0: ≤120 mg/dl, 1: >120 mg/dl)</li>
        <li>restecg: Hasil EKG istirahat (0: normal, 1: ST-T abnormal, 2: pembesaran ventrikel)</li>
        <li>Thalach: Denyut jantung maksimum</li>
        <li>Exang: Angina saat olahraga (0: tidak, 1: ya)</li>
        <li>Oldpeak: Depresi ST akibat olahraga</li>
        <li>slope: Kemiringan segmen ST (0: menanjak, 1: datar, 2: menurun)</li>
        <li>ca: Jumlah pembuluh darah utama (0-3)</li>
        <li>thal: Hasil tes thalium (1: normal, 2: fixed defect, 3: reversable defect)</li>
    </ul>
    <p>Deteksi dini dapat membantu pencegahan dan pengobatan lebih efektif.</p>
</div>
""", unsafe_allow_html=True)

# Form input pengguna
st.markdown('<div class="header">Masukkan data pasien untuk prediksi risiko penyakit jantung:</div>', unsafe_allow_html=True)

# Input fitur
# Dua kolom input
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Usia", 0, 100, 50)
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    cp = st.selectbox("Tipe Nyeri Dada",  options=[0, 1, 2, 3], format_func=lambda x: ["Tidak Nyeri", "Ringan", "Sedang", "Parah"][x])
    trestbps = st.number_input("Tekanan Darah Istirahat (mmHg)", 90, 300, 120)
with col2:
    chol = st.number_input("Kolesterol (mg/dL)", 100, 600, 200)
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", ["Tidak", "Ya"])
    restecg = st.selectbox("Hasil EKG Istirahat", [0, 1, 2])
    thalach = st.number_input("Detak Jantung Maksimum", 70, 210, 150)
with col3:
    exang = st.selectbox("Angina Induksi Olahraga?", ["Tidak", "Ya"])
    oldpeak = st.slider("Depresi ST (Oldpeak)", -2.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("Kemiringan ST", options=[0, 1, 2], format_func=lambda x: ["Menanjak", "Datar", "Menurun"][x])
    ca = st.selectbox("Jumlah Pembuluh Darah yang Utama (0-3)", [0, 1, 2, 3])
    thal_label = st.selectbox(
    "Hasil Tes Thalium (Thalassemia)",
    ["Normal", "Fixed Defect", "Reversable Defect"])
    thal_dict = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversable Defect": 3}
    thal = thal_dict[thal_label]
    
# Ubah input menjadi array
input_data = np.array([[
    age,
    1 if sex == "Laki-laki" else 0,
    cp,
    trestbps,
    chol,
    1 if fbs == "Ya" else 0,
    restecg,
    thalach,
    1 if exang == "Ya" else 0,
    oldpeak,
    slope,
    ca,
    thal
]])

# Normalisasi
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_scaled)
    pred_label = int(prediction[0][0] > 0.5)
    st.subheader("Hasil Prediksi")
    if pred_label == 1:
        st.error("Disease")
    else:
        st.success("Non-Disease.")
    

# Footer
st.markdown("---")
st.caption("© 2025 Rizka Dwi Arzita | Skripsi - Identifikasi Penyakit Jantung dengan Model MLP yang dioptimasi PSO")
