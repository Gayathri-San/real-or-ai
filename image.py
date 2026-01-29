import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

st.set_page_config(page_title="CNN Image Classifier", layout="centered")

st.title("CNN Image Classification App")

st.write("Upload an image and get prediction from CNN model")

@st.cache_resource

def load_model():
    model = tf.keras.models.load_model("cnn_model.keras")
    return model

model = load_model()


class_names = ["Class 0", "Class 1"]

def preprocess_image(image):
    image = image.resize((100, 100))   # same size used in training
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess_image(image)

if st.button("Predict"):
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: **{class_names[predicted_class]}**")

