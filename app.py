import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# KONFIGURASI
# ===============================
st.set_page_config(
    page_title="Shark Species Identifier",
    layout="centered"
)

MODEL_PATH = "shark_cnn_model.h5"

CLASS_NAMES = [
    "tiger_shark",
    "whale_shark",
    "white_shark",
    "whitetip_shark",
    # lanjutkan sampai 14 kelas
    # URUTANNYA HARUS SAMA DENGAN TRAINING
]

IMG_SIZE = (224, 224)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# UI
# ===============================
st.title("🦈 Shark Species Identifier (CNN)")
st.write("Upload foto hiu, sistem akan memprediksi spesiesnya.")

uploaded_file = st.file_uploader(
    "Upload gambar hiu",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# PREDIKSI
# ===============================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    with st.spinner("Menganalisis gambar..."):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = predictions[0][predicted_index]

    st.success("Prediksi selesai!")

    st.markdown(f"""
    ### 🦈 Hasil Prediksi:
    **Spesies:** `{CLASS_NAMES[predicted_index]}`  
    **Confidence:** `{confidence*100:.2f}%`
    """)

    # ===============================
    # TOP 3 PREDICTION
    # ===============================
    st.write("### 🔍 Top 3 Prediksi:")
    top_3_idx = np.argsort(predictions[0])[::-1][:3]

    for i in top_3_idx:
        st.write(
            f"- {CLASS_NAMES[i]} : {predictions[0][i]*100:.2f}%"
        )
