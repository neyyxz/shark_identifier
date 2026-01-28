import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "shark_model.tflite"

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "basking_shark",
    "blacktip_shark",
    "blue_shark",
    "bull_shark",
    "hammerhead_shark",
    "lemon_shark",
    "mako_shark",
    "nurse_shark",
    "sand_tiger_shark",
    "thresher_shark",
    "tiger_shark",
    "whale_shark",
    "white_shark",
    "whitetip_shark",
]

st.title("🦈 Shark Species Identifier")
st.write("Upload foto hiu, sistem akan memprediksi spesiesnya.")
st.write("Sistem ini menggunakan model berbasis CNN (Convolutional Neural Network) yang sudah dilatih dan memiliki tingkat akuarasi tinggi")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])

    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {CLASS_NAMES[class_idx]}")
    st.write(f"Confidence: {confidence:.2%}")

