import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime

# --- PENGATURAN AWAL ---
st.set_page_config(page_title="Absensi Wajah Otomatis", layout="wide")
st.title("📸 Sistem Absensi Otomatis Berbasis Pengenalan Wajah")
st.write("Aplikasi ini akan mendeteksi wajah Anda dan mencatat kehadiran secara otomatis.")

# --- MEMUAT MODEL & LABEL ---
# Muat model Keras yang sudah dilatih dari Teachable Machine
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    return model

# Muat label nama dari file labels.txt
def load_labels():
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

model = load_my_model()
class_names = load_labels()

# --- INISIALISASI DATA ABSENSI ---
# Gunakan st.session_state untuk menyimpan data absensi agar tidak hilang saat aplikasi di-refresh
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = pd.DataFrame(columns=["Nama", "Waktu Absen"])


# --- FUNGSI UTAMA UNTUK PENGENALAN WAJAH ---
def process_and_predict(img):
    # 1. Pra-pemrosesan gambar sesuai dengan yang dibutuhkan model Teachable Machine
    image = np.array(img)
    image = cv2.resize(image, (224, 224)) # Sesuaikan ukuran gambar
    image_normalized = (image.astype(np.float32) / 127.5) - 1 # Normalisasi
    image_reshaped = np.expand_dims(image_normalized, axis=0) # Tambah dimensi batch

    # 2. Lakukan Prediksi
    prediction = model.predict(image_reshaped)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 3. Kembalikan hasil prediksi
    return class_name.strip(), confidence_score


# --- TATA LETAK APLIKASI ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Ambil Gambar")
    # Menggunakan st.camera_input untuk mengambil gambar dari webcam
    img_file_buffer = st.camera_input("Klik untuk mengambil gambar dan melakukan absensi...")

    if img_file_buffer:
        # Konversi buffer gambar ke format yang bisa dibaca PIL dan OpenCV
        img = Image.open(img_file_buffer)

        # Proses gambar dan dapatkan prediksi
        predicted_name, confidence = process_and_predict(img)

        # Tampilkan gambar yang diambil
        st.image(img, caption="Gambar yang Diambil", use_column_width=True)

        # Tampilkan hasil prediksi jika confidence score cukup tinggi
        st.subheader("Hasil Prediksi:")
        if confidence > 0.90:  # Anda bisa menyesuaikan ambang batas ini (misal: 90%)
            st.success(f"Wajah terdeteksi sebagai: **{predicted_name}**")
            st.info(f"Tingkat Keyakinan: {confidence:.2%}")

            # Catat Kehadiran
            current_time = datetime.now().strftime("%H:%M:%S")
            new_entry = pd.DataFrame([[predicted_name, current_time]], columns=["Nama", "Waktu Absen"])
            
            # Cek apakah nama sudah ada di daftar absensi hari ini
            if predicted_name not in st.session_state.attendance_df['Nama'].values:
                st.session_state.attendance_df = pd.concat([st.session_state.attendance_df, new_entry], ignore_index=True)
                st.balloons()
                st.success(f"✅ Absensi untuk **{predicted_name}** berhasil dicatat!")
            else:
                st.warning(f"⚠️ **{predicted_name}** sudah melakukan absensi sebelumnya.")

        else:
            st.error("Wajah tidak dikenali atau tingkat keyakinan terlalu rendah. Silakan coba lagi.")


with col2:
    st.header("2. Daftar Hadir Hari Ini")
    # Gunakan st.empty() agar tabel bisa di-update secara dinamis
    attendance_table = st.empty()
    attendance_table.dataframe(st.session_state.attendance_df, use_container_width=True)

    # Tombol untuk mereset daftar hadir
    if st.button("Reset Daftar Hadir"):
        st.session_state.attendance_df = pd.DataFrame(columns=["Nama", "Waktu Absen"])
        attendance_table.dataframe(st.session_state.attendance_df, use_container_width=True)
        st.info("Daftar hadir telah direset.")