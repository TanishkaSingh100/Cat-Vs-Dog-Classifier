import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Set page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")


# Header section
st.title("🐾 Cat vs Dog Classifier")
st.markdown("Upload an image of a Cat or Dog, and let our model do the magic 🪄")

# Toggle Mode: Fun vs Basic
mode = st.radio("🎮 Choose App Mode:", ["Fun Mode ", "Basic Mode "])
show_fun = (mode == "Fun Mode ")

# Sidebar: Model Info
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🔍 About the Model</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:15px; line-height:1.6'>
    <ul>
        <li><strong>Type:</strong> Convolutional Neural Network </li>
        <li><strong>Framework:</strong> TensorFlow / Keras </li>
        <li><strong>Input Size:</strong> 256×256×3 </li>
        <li><strong>Output:</strong> Cat or Dog </li>
        <li><strong>Final Activation:</strong> Sigmoid </li>
        <li><strong>Optimizer:</strong> Adam </li>
        <li><strong>Accuracy:</strong> ~80% </li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:13px;'>✨ Built by <b>Tanishka</b> ✨</div>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("🖼 Upload Image", type=["jpg", "jpeg", "png"])

# Load model
@st.cache_resource
def load_model():
    model_path = "cat_dog_model.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=11cT5PdcMLsr5p125JByo0YQy1zKgNSuC"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()


# Define confidence bar function (scale 0-5)
def confidence_bar(conf):
    scale = int(conf * 5)  # if conf is 0.66, scale = 3 (range: 0-5)
    emojis = "😿😾😼😺😸"
    return emojis[:scale + 1]

# Prediction function
def predict(image):
    image = image.resize((256, 256))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Process the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    
    label, confidence = predict(image)
    
    if show_fun:
        # Colored result box
        if "Dog" in label:
            st.markdown(f"<div style='background-color:#d0f0fd; padding:20px; border-radius:12px; text-align:center;'>"
                    f"<h2 style='color:#034694;'>🐶 It's a Dog! 🐶</h2>"
                    f"<p style='font-size:18px; color:#034694;'>Confidence: <strong>{confidence:.2%}</strong></p>"
                    f"<p style='font-size:24px;'>{confidence_bar(confidence)}</p>"
                    f"</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#ffe4f0; padding:20px; border-radius:12px; text-align:center;'>"
                    f"<h2 style='color:#c71585;'>🐱 It's a Cat! 🐱</h2>"
                    f"<p style='font-size:18px; color:#c71585;'>Confidence: <strong>{confidence:.2%}</strong></p>"
                    f"<p style='font-size:24px;'>{confidence_bar(confidence)}</p>"
                    f"</div>", unsafe_allow_html=True)

        # Optional: Progress bar (visually clean)
        st.progress(float(confidence))


    else:
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2%}")
