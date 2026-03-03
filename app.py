import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AgroAI - Smart Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
}
.big-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #1b5e20;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #2e7d32;
}
.result-card {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
}
.footer {
    text-align:center;
    color:gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = tf.keras.models.load_model("plant_disease_model.h5")

with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = sorted(class_indices, key=class_indices.get)

# ---------------- REMEDIES ---------------- #
remedies = {
    "Tomato___Early_blight": "Remove infected leaves. Apply fungicide spray weekly.",
    "Tomato___Late_blight": "Avoid overhead watering. Use copper-based fungicide.",
    "Potato___Healthy": "Plant is healthy. Maintain proper irrigation & monitoring.",
    "Potato___Early_blight": "Practice crop rotation & use recommended fungicides."
}

# ---------------- HEADER ---------------- #
st.markdown('<div class="big-title">🌿 AgroAI Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered plant disease detection for smarter farming</div>', unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🌾 AgroAI Dashboard")
st.sidebar.write("Upload leaf images to detect plant diseases instantly.")
st.sidebar.success("Model Accuracy: ~90%")
st.sidebar.info("Built using CNN + TensorFlow")

# ---------------- FILE UPLOAD ---------------- #
file = st.file_uploader("📸 Upload Leaf Image", type=["jpg","png"])

if file:
    col1, col2 = st.columns([1,1])

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index])
    disease = class_names[class_index]

    # LEFT SIDE - IMAGE
    with col1:
        st.image(file, caption="Uploaded Leaf Image", width="stretch")

    # RIGHT SIDE - RESULT CARD
    with col2:
       
        st.markdown("### 🔍 Detection Result")

        if "Healthy" in disease:
            st.success(f"🌱 {disease}")
            severity = "Low"
        else:
            st.error(f"⚠ {disease}")
            severity = "Moderate"

        st.markdown("### 📊 Confidence Score")
        st.progress(confidence)
        st.write(f"{round(confidence*100,2)} %")

        st.markdown("### 🌡 Severity Level")
        st.info(severity)

        st.markdown("### 💡 Recommended Action")
        st.write(remedies.get(disease, "No remedy available."))

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("---")

st.markdown('<div class="footer">© 2026 AgroAI | Built by Varsha 🌿</div>', unsafe_allow_html=True)