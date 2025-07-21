import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import pandas as pd

# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Kematangan Lemon",
    page_icon="🍋",
    layout="wide"
)

# --- FUNGSI-FUNGSI ---

# Cache untuk mempercepat loading model
@st.cache_resource
def load_keras_model(model_path):
    """Memuat model Keras dari file .h5 dan meng-cache-nya."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

def preprocess_image(image, target_size=(150, 200)):
    """Memproses gambar yang diunggah untuk prediksi."""
    img = Image.open(image).convert("RGB")  # Convert to RGB to ensure 3 channels
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# --- APLIKASI UTAMA ---

# Judul dan Deskripsi Aplikasi
st.title("🍋 Aplikasi Klasifikasi Kematangan Lemon")
st.write(
    "Unggah gambar buah lemon dan aplikasi ini akan memprediksi tingkat kematangannya "
    "menggunakan model Deep Learning yang telah dilatih."
)

MODEL_PATH = 'lemon_maturity_classifier_final.h5'
model = load_keras_model(MODEL_PATH)

CLASS_NAMES = ['.', 'matang', 'mentah'] # Sesuaikan jika ada lebih dari 2 kelas atau urutannya berbeda

# Cek apakah model berhasil dimuat
if model is None:
    st.warning("Model tidak dapat dimuat. Pastikan file model ada di direktori yang sama dengan 'app.py'.")
else:
    # Komponen untuk upload file
    uploaded_file = st.file_uploader(
        "Pilih gambar lemon...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gambar yang Diunggah")
            # Pra-pemrosesan gambar
            processed_image, original_image = preprocess_image(uploaded_file)
            st.image(original_image, caption="Gambar Lemon", use_column_width=True)

        with col2:
            st.subheader("Hasil Prediksi")
            with st.spinner("Model sedang menganalisis..."):
                predictions = model.predict(processed_image)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_idx]

            # Menampilkan hasil dengan lebih menarik
            if predicted_class.lower() == 'matang':
                st.success(f"**Prediksi: {predicted_class.capitalize()}** 🟢")
            elif predicted_class.lower() == 'mentah':
                st.warning(f"**Prediksi: {predicted_class.capitalize()}** 🟠")
            else: # Tambahkan penanganan jika ada kelas lain atau tidak teridentifikasi
                st.info(f"**Prediksi: {predicted_class.capitalize()}**")
            
            st.markdown(f"**Tingkat Keyakinan:** `{confidence:.2%}`")

            # Membuat bar chart untuk probabilitas
            st.write("---") # Garis pemisah
            st.subheader("Distribusi Probabilitas")
            
            # Ubah data probabilitas ke dalam format yang sesuai untuk st.bar_chart
            prob_df = pd.DataFrame({
                'Kelas': CLASS_NAMES,
                'Probabilitas': predictions[0]
            })
            prob_df = prob_df.set_index('Kelas')

            st.bar_chart(prob_df)


### Informasi Kelompok Pembuat

st.markdown("""
---
Aplikasi ini dikembangkan sebagai tugas dari Mata Kuliah Sistem Cerdas:

* **Aditya Zhafari Nur Itmam**
* **Muhammad Ardiansyah Firdaus**
* **Naufal Zahran Suhartono**
""")