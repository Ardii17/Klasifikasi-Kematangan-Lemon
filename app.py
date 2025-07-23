import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Kematangan Lemon",
    page_icon="🍋",
    layout="wide"
)

# Path ke model (harus berada di folder yang sama)
MODEL_PATH = "lemon_maturity_classifier_final.h5"

# Daftar kelas yang dikenali model
CLASS_NAMES = ['.', 'matang', 'mentah']  # Sesuaikan urutan ini jika perlu

# Fungsi load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Fungsi pra-pemrosesan gambar
def preprocess_image(uploaded_file, target_size=(150, 200)):
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image

# Judul aplikasi
st.title("🍋 Aplikasi Klasifikasi Kematangan Lemon")
st.write(
    "Unggah gambar lemon, dan aplikasi ini akan memprediksi apakah buah tersebut sudah matang atau belum menggunakan model deep learning."
)

# Load model
model = load_model()

if model is None:
    st.stop()

# Komponen Upload
uploaded_file = st.file_uploader("Pilih gambar lemon...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar yang Diupload")
        processed_image, original_image = preprocess_image(uploaded_file)
        st.image(original_image, caption="Gambar Lemon", use_column_width=True)

    with col2:
        st.subheader("Hasil Prediksi")
        with st.spinner("Model sedang memproses gambar..."):
            predictions = model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]

        if predicted_class.lower() == 'matang':
            st.success(f"**Prediksi: {predicted_class.capitalize()}** 🟢")
        elif predicted_class.lower() == 'mentah':
            st.warning(f"**Prediksi: {predicted_class.capitalize()}** 🟠")
        else:
            st.info(f"**Prediksi: {predicted_class.capitalize()}**")

        st.markdown(f"**Tingkat Keyakinan:** `{confidence:.2%}`")

        # Visualisasi Distribusi
        st.write("---")
        st.subheader("Distribusi Probabilitas")
        prob_df = pd.DataFrame({
            'Kelas': CLASS_NAMES,
            'Probabilitas': predictions[0]
        }).set_index('Kelas')
        st.bar_chart(prob_df)

# Informasi Pembuat
st.markdown("""---  
Aplikasi ini dikembangkan dalam rangka tugas Mata Kuliah Sistem Cerdas oleh:

- **Aditya Zhafari Nur Itmam**  
- **Muhammad Ardiansyah Firdaus**  
- **Naufal Zahran Suhartono**
""")
