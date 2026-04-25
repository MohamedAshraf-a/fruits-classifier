import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Fruits Classifier",
    page_icon="🍎",
    layout="centered"
)

# --- 2. CSS Styling ---
st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #1b5e20; }
    .result-header { text-align: center; color: #1e3d59; }
</style>
""", unsafe_allow_html=True)

# --- 3. Load Model & Labels ---

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('fruits_model.h5')

    if os.path.exists('labels.txt'):
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        labels = []

    return model, labels

model, labels = load_resources()

# --- 4. Sidebar ---
with st.sidebar:
    st.title("About Project")
    st.markdown("---")
    st.write("👨‍💻 Abdelrahman Atef")
    st.write("👨‍💻 Mohamed Ashraf")
    st.markdown("---")
    st.write("Model: MobileNetV2")
    st.write("Task: Fruit Classification")

# --- 5. Main UI ---
st.markdown("<h1 class='result-header'>🍎 AI Fruits Classifier</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict 🔍"):

        with st.spinner("AI is analyzing..."):

            # 🔥 FIXED PREPROCESSING (IMPORTANT)
            img_resized = image.resize((224, 224)).convert('RGB')
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array)

            score = np.max(predictions)
            index = np.argmax(predictions)

            # Threshold
            THRESHOLD = 0.7

            st.markdown("---")

            if score < THRESHOLD:
                st.error("⚠️ Unknown Object")
                st.warning(f"Confidence too low: {score*100:.2f}%")
            else:
                label = labels[index] if labels else str(index)
                label = label.replace("_", " ").title()

                is_rotten = "rotten" in label.lower()

                if is_rotten:
                    st.error(f"🍂 {label}")
                    st.info("Status: Rotten / Not Fresh")
                else:
                    st.success(f"🍎 {label}")
                    st.info("Status: Fresh / Edible")

                st.write(f"Confidence: {score*100:.2f}%")

else:
    st.info("Upload an image to start prediction")

# --- 6. Footer ---
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'> 22262 _  22067</p>
""", unsafe_allow_html=True)