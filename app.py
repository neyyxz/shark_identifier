import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================
# LOAD LABELS
# =====================
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# =====================
# LOAD TFLITE MODEL
# =====================
interpreter = tf.lite.Interpreter(model_path="shark_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

# =====================
# STREAMLIT UI
# =====================
st.title("🦈 Shark Species Identifier")
st.write("Upload foto hiu, model akan memprediksi spesiesnya")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"🦈 Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
