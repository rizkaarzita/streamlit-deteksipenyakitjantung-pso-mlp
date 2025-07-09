# ❤️ Deteksi Penyakit Jantung dengan MLP-PSO (Streamlit App)

Proyek ini merupakan aplikasi berbasis web untuk mendeteksi risiko penyakit jantung menggunakan model **Multilayer Perceptron (MLP)** yang telah dioptimasi menggunakan algoritma **Particle Swarm Optimization (PSO)**. Aplikasi ini dibangun menggunakan **Streamlit** dan dapat dijalankan secara lokal atau di-hosting melalui [Streamlit Cloud](https://streamlit.io/).

## 🚀 Fitur Aplikasi

- Input data pasien secara manual melalui antarmuka interaktif.
- Normalisasi input menggunakan `MinMaxScaler` dari `scikit-learn`.
- Prediksi risiko penyakit jantung menggunakan model MLP-PSO yang sudah dilatih.
- Visualisasi hasil prediksi dengan output **Disease** atau **Non-Disease**.
- Antarmuka pengguna yang intuitif dan mudah digunakan.

## 🧠 Tentang Model

Model yang digunakan adalah hasil pelatihan **Multilayer Perceptron (MLP)** dengan optimasi hyperparameter menggunakan algoritma **PSO (Particle Swarm Optimization)**. Model disimpan dalam format `SavedModel` agar kompatibel dengan Streamlit Cloud.

## 🗂️ Struktur File
```bash
streamlit-deteksipenyakitjantung-pso-mlp/
├── app_penyakit_jantung.py # File utama Streamlit
├── scaler.save # Objek scaler (MinMaxScaler)
├── model_mlp_pso_savedmodel.zip # Model terlatih dalam format SavedModel (zip)
├── requirements.txt # Daftar dependensi Python
├── README.md # Dokumentasi (file ini)
```
📊 Dataset
Model ini dilatih menggunakan dataset Heart Disease UCI Machine Learning yang sudah diproses sebelumnya.

