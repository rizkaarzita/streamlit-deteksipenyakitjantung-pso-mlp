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
# model = tf.keras.models.load_model("model_mlp_pso.h5")
# model = tf.keras.models.load_model("model_mlp_pso.keras")

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

# Sidebar: Informasi tambahan
st.sidebar.markdown("""
<div style="background-color:#f9f9f9; padding:15px; border-radius:10px; font-size:14px; line-height:1.6;">
    <h3 style="color:#d62828;">📋 Informasi Variabel Penelitian</h3>
    <ul style="padding-left:20px;">
        <li><b>age:</b> usia pasien</li>
        <li><b>sex:</b> jenis kelamin <br>
            0: Perempuan, <br>
            1: Laki-laki
        </li>
        <li><b>cp:</b> jenis nyeri dada <br>
            typical angina: nyeri dada yang memiliki gejala biasa, <br>
            atypical angina: nyeri dada yang tidak bisa diprediksi, <br>
            non-anginal pain: gejala di luar penyakit jantung, <br>
            asymptomatic: tanpa gelaja
        </li>
        <li><b>trestbps:</b> Tekanan darah saat istirahat (mm Hg)</li>
        <li><b>chol:</b> Kadar kolesterol (mg/dl)</li>
        <li><b>fbs:</b> Gula darah puasa <br>
            0: ≤120 mg/dl, <br>
            1: >120 mg/dl
        </li>
        <li><b>restecg:</b> Hasil EKG istirahat <br>
            0: normal, <br>
            1: ST-T abnormal, <br>
            2: kondisi saat ventricular kiri mengalami hipertrofi
        </li>
        <li><b>Thalach:</b> Denyut jantung maksimum</li>
        <li><b>Exang:</b> Keadaan pasien akan mengalami nyeri dada apabila berolahraga <br>
            0: tidak nyeri, <br>
            1: menyebabkan nyeri
        </li>
        <li><b>Oldpeak:</b> Penurunan segmen ST pada EKG disebabkan oleh olahraga</li>
        <li><b>slope:</b> Kemiringan segmen ST pada EKG setelah berolahraga <br>
            Upsloping: detak jantung yang lebih baik dengan olahraga, <br>
            Flatsloping: jantung sehat yang khas, <br>
            Downsloping: tanda-tanda jantung yang tidak sehat
        </li>
        <li><b>ca:</b> Jumlah pembuluh darah utama <br>
            0: tidak ada penyumbatan, <br>
            1: satu pembuluh tersumbat, <br>
            2: dua pembuluh tersumbat, <br>
            3: tiga pembuluh tersumbat
        </li>
        <li><b>thal:</b> Hasil tes thalium <br>
            normal, <br>
            fixed defect: terdapat bagian jantung yang permanen rusak (jaringan jantung sudah tidak berfungsi normal), <br>
            reversable defect: terdapat gangguan aliran darah ke jantung, tapi sifatnya sementara dan bisa membaik setelah istirahat atau pengobatan
        </li>
    </ul>
    <p>Deteksi dini dapat membantu pencegahan dan pengobatan lebih efektif.</p>
</div>
""", unsafe_allow_html=True)

# # Sidebar: Informasi tambahan di sidebar
# st.sidebar.markdown("""
# <div class="info-box">
#     <h3>Informasi Penyakit Jantung</h3>
#     <p>Penyakit jantung adalah salah satu penyebab kematian utama di dunia. Berikut variabel penelitian:</p>
#     <ul>
#         <li>age: usia pasien</li>
#         <li>sex: jenis kelamin (0: Perempuan, 1: Laki-laki)</li>
#         <li>cp: jenis nyeri dada (typical angina: nyeri dada yang memiliki gejala biasa, atypical angina: nyeri dada yang tidak bisa diprediksi, non-anginal pain: gejala di luar penyakit jantung, asymptomatic: tanpa gelaja)</li>
#         <li>trestbps: Tekanan darah saat istirahat (mm Hg)</li>
#         <li>chol: Kadar kolesterol (mg/dl) (>200 berisiko)</li>
#         <li>fbs: Gula darah puasa (0: ≤120 mg/dl, 1: >120 mg/dl)</li>
#         <li>restecg: Hasil EKG istirahat (0: normal, 1: ST-T abnormal, 2: kondisi saat ventricular kiri mengalami hipertropi)</li>
#         <li>Thalach: Denyut jantung maksimum</li>
#         <li>Exang: Keadaan pasien akan mengalami nyeri dada apabila berolahraga (0: tidak nyeri, 1: menyebabkan nyeri)</li>
#         <li>Oldpeak: Penurunan segmen ST pada EKG disebabkan oleh olahraga </li>
#         <li>slope: Kemiringan segmen ST pada EKG setelah berolahraga (Upsloping: detak jantung yang lebih baik dengan olahraga, Flatsloping: jantung sehat yang khas, Downsloping: tanda-tanda jantung yang tidak sehat)</li>
#         <li>ca: Jumlah pembuluh darah utama (0: tidak ada penyumbatan, 1: satu pembuluh tersumbat, 2: dua pembuluh tersumbat, 3: tiga pembuluh tersumbat)</li>
#         <li>thal: Hasil tes thalium (normal, fixed defect: terdapat bagian jantung yang permanen rusak (jaringan jantung sudah tidak berfungsi normal) , reversable defect: terdapat gangguan aliran darah ke jantung, tapi sifatnya sementara dan bisa membaik setelah istirahat atau pengobatan)</li>
#     </ul>
#     <p>Deteksi dini dapat membantu pencegahan dan pengobatan lebih efektif.</p>
# </div>
# """, unsafe_allow_html=True)

# Form input pengguna
st.markdown('<div class="header">Masukkan data pasien untuk prediksi risiko penyakit jantung:</div>', unsafe_allow_html=True)

# Input fitur
# Dua kolom input
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Usia", 0, 100, 50)
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    cp = st.selectbox("Tipe Nyeri Dada",  options=[0, 1, 2, 3], format_func=lambda x: ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"][x])
    trestbps = st.number_input("Tekanan Darah Istirahat (mmHg)", 90, 300, 120)
with col2:
    chol = st.number_input("Kolesterol (mg/dL)", 100, 600, 200)
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", ["Tidak", "Ya"])
    restecg = st.selectbox("Hasil EKG Istirahat", options=[0, 1, 2], format_func=lambda x: ["normal", "kondisi ST-T wave abnormality", "ventricular kiri mengalami hipertropi"][x])
    thalach = st.number_input("Detak Jantung Maksimum", 70, 210, 150)
with col3:
    exang = st.selectbox("Angina Induksi Olahraga?", ["Tidak nyeri", "Menyebabkan nyeri"])
    oldpeak = st.slider("Depresi ST (Oldpeak)", -2.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("Kemiringan ST", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flatsloping", "Downsloping"][x])
    ca = st.selectbox("Jumlah Pembuluh Darah yang tersumbat atau mengalami gangguan (0-3)", [0, 1, 2, 3])
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
    # Gunakan fungsi dari signature SavedModel
    predict_fn = model.signatures["serving_default"]
    
    # Konversi ke tensor
    input_tensor = tf.convert_to_tensor(input_scaled, dtype=tf.float32)

    # Lakukan prediksi
    prediction_dict = predict_fn(input_tensor)

    # Ambil nilai prediksi dari dictionary
    prediction = list(prediction_dict.values())[0].numpy()

    # Klasifikasi biner
    pred_label = int(prediction[0][0] > 0.5)
    st.subheader("Hasil Prediksi")
    if pred_label == 1:
        st.error("Memiliki risiko penyakit jantung")
    else:
        st.success("Tidak memiliki risiko penyakit jantung")

# if st.button("Prediksi"):
#     prediction = model.predict(input_scaled)
#     pred_label = int(prediction[0][0] > 0.5)
#     st.subheader("Hasil Prediksi")
#     if pred_label == 1:
#         st.error("Disease")
#     else:
#         st.success("Non-Disease.")
    

# Footer
st.markdown("---")
st.caption("© 2025 Rizka Dwi Arzita | Skripsi - Identifikasi Penyakit Jantung dengan Model MLP yang dioptimasi PSO")



