import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================
# LOAD MODEL
# =====================
MODEL_PATH = "shark_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# CLASS LABELS
# ⚠️ HARUS SAMA PERSIS DENGAN HASIL train_data.class_indices
# =====================
CLASS_NAMES = [
    "tiger_shark",
    "whale_shark",
    "white_shark",
    "whitetip_shark"
]

# =====================
# UI
# =====================
st.set_page_config(page_title="Shark Identifier", page_icon="🦈")
st.title("🦈 Shark Species Identifier")
st.write("Upload gambar hiu, model akan mendeteksi spesiesnya.")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

# =====================
# PREDICTION FUNCTION
# =====================
def predict(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    return CLASS_NAMES[class_index], confidence

# =====================
# RUN
# =====================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, conf = predict(image)

    st.markdown(f"### 🦈 Prediksi: **{label.replace('_', ' ').title()}**")
    st.markdown(f"### 🔍 Confidence: **{conf*100:.2f}%**")
