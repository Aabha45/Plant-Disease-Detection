import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.card {
    background-color: #f5f7f6;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model('models/model.h5',compile=False)

# ===============================
# LOAD CLASS LABELS
# ===============================
with open('models/class_indices.json') as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# ===============================
# DISEASE DATABASE
# ===============================
disease_info = {

# 🍎 APPLE
"Apple___Apple_scab": {
    "cause": "Fungal infection caused by Venturia inaequalis.",
    "prevention": [
        "Prune trees to improve air circulation.",
        "Apply fungicides during early stages.",
        "Remove fallen infected leaves.",
        "Use resistant apple varieties.",
        "Avoid overhead watering."
    ]
},

# 🫐 BLUEBERRY
"Blueberry___healthy": {
    "cause": "No disease detected.",
    "prevention": [
        "Maintain proper soil acidity.",
        "Water plants regularly.",
        "Ensure good sunlight exposure.",
        "Use proper fertilizers.",
        "Monitor plant health regularly."
    ]
},

# 🍒 CHERRY
"Cherry___healthy": {
    "cause": "No disease.",
    "prevention": [
        "Maintain orchard cleanliness.",
        "Ensure proper irrigation.",
        "Monitor for pests regularly.",
        "Use balanced fertilizers.",
        "Provide adequate sunlight."
    ]
},

# 🌽 CORN
"Corn___Commonrust": {
    "cause": "Fungal infection spread by spores.",
    "prevention": [
        "Use resistant corn hybrids.",
        "Apply fungicides if needed.",
        "Practice crop rotation.",
        "Monitor crops regularly.",
        "Maintain plant nutrition."
    ]
},
"Corn___Grayleafspot": {
    "cause": "Fungal disease in humid conditions.",
    "prevention": [
        "Rotate crops regularly.",
        "Avoid excess irrigation.",
        "Use resistant varieties.",
        "Apply fungicides.",
        "Remove infected debris."
    ]
},
"Corn___LeafBlight": {
    "cause": "Fungal infection due to moisture.",
    "prevention": [
        "Ensure good drainage.",
        "Use quality seeds.",
        "Apply fungicides early.",
        "Avoid overcrowding.",
        "Practice crop rotation."
    ]
},
"Corn___healthy": {
    "cause": "No disease.",
    "prevention": [
        "Maintain soil fertility.",
        "Ensure proper watering.",
        "Monitor crops regularly.",
        "Control weeds.",
        "Use quality seeds."
    ]
},

# 🍇 GRAPE
"Grape___Black_rot": {
    "cause": "Fungal infection affecting leaves and fruit.",
    "prevention": [
        "Remove infected parts.",
        "Apply fungicides regularly.",
        "Ensure proper airflow.",
        "Avoid overhead watering.",
        "Prune vines properly."
    ]
},
"Grape___healthy": {
    "cause": "No disease.",
    "prevention": [
        "Maintain vineyard hygiene.",
        "Regular pruning.",
        "Balanced fertilization.",
        "Monitor plant health.",
        "Ensure sunlight exposure."
    ]
},

# 🌶️ PEPPER
"PepperBell___Bacterialspot": {
    "cause": "Bacterial infection spread by water.",
    "prevention": [
        "Avoid overhead watering.",
        "Use disease-free seeds.",
        "Remove infected leaves.",
        "Sterilize tools.",
        "Apply copper sprays."
    ]
},
"PepperBell___healthy": {
    "cause": "No disease.",
    "prevention": [
        "Maintain proper irrigation.",
        "Use balanced fertilizers.",
        "Inspect plants regularly.",
        "Ensure sunlight exposure.",
        "Keep field clean."
    ]
},

# 🥔 POTATO
"Potato___Early_blight": {
    "cause": "Fungal infection caused by Alternaria.",
    "prevention": [
        "Practice crop rotation.",
        "Apply fungicides early.",
        "Remove infected leaves.",
        "Avoid overwatering.",
        "Maintain soil health."
    ]
},
"Potato___Late_blight": {
    "cause": "Fungal-like organism in humid conditions.",
    "prevention": [
        "Avoid water accumulation.",
        "Ensure proper drainage.",
        "Use certified seeds.",
        "Apply fungicides.",
        "Remove infected plants."
    ]
},

# 🌱 SOYBEAN
"Soyabean___healthy": {
    "cause": "No disease.",
    "prevention": [
        "Use quality seeds.",
        "Maintain soil fertility.",
        "Monitor regularly.",
        "Ensure proper irrigation.",
        "Control pests."
    ]
},

# 🎃 SQUASH
"Squash___Powdery_mildew": {
    "cause": "Fungal infection in dry conditions.",
    "prevention": [
        "Ensure good air circulation.",
        "Avoid overcrowding.",
        "Apply fungicides.",
        "Water at base.",
        "Remove infected leaves."
    ]
},

# 🍓 STRAWBERRY
"Strawberry___healthy": {
    "cause": "No disease.",
    "prevention": [
        "Maintain clean soil.",
        "Avoid overwatering.",
        "Ensure sunlight exposure.",
        "Use mulch.",
        "Monitor regularly."
    ]
},

# 🍅 TOMATO
"Tomato___Bacterial_spot": {
    "cause": "Bacterial infection.",
    "prevention": [
        "Avoid overhead watering.",
        "Use disease-free seeds.",
        "Remove infected leaves.",
        "Disinfect tools.",
        "Apply bactericides."
    ]
},
"Tomato___Early_blight": {
    "cause": "Fungal infection.",
    "prevention": [
        "Crop rotation.",
        "Remove infected debris.",
        "Apply fungicides.",
        "Avoid excess watering.",
        "Maintain nutrition."
    ]
},
"Tomato___Late_blight": {
    "cause": "Fungal-like organism in humid weather.",
    "prevention": [
        "Avoid overhead watering.",
        "Ensure good airflow.",
        "Apply fungicides.",
        "Remove infected plants.",
        "Use resistant varieties."
    ]
},
"Tomato___LeafMold": {
    "cause": "Fungal growth in humid environment.",
    "prevention": [
        "Reduce humidity.",
        "Increase ventilation.",
        "Avoid leaf wetness.",
        "Use resistant varieties.",
        "Apply fungicides."
    ]
},
"Tomato___mosaicvirus": {
    "cause": "Viral infection spread by contact.",
    "prevention": [
        "Use virus-free seeds.",
        "Disinfect tools.",
        "Remove infected plants.",
        "Avoid touching plants frequently.",
        "Control pests."
    ]
},
"Tomato___Septorialeafspot": {
    "cause": "Fungal infection in wet conditions.",
    "prevention": [
        "Avoid overhead watering.",
        "Remove infected leaves.",
        "Apply fungicides.",
        "Ensure proper spacing.",
        "Rotate crops."
    ]
},
"Tomato___YellowLeafCurlVirus": {
    "cause": "Virus spread by whiteflies.",
    "prevention": [
        "Control whiteflies.",
        "Use resistant varieties.",
        "Install insect nets.",
        "Remove infected plants.",
        "Avoid nearby infected crops."
    ]
},
"Tomato___healthy": {
    "cause": "No disease.",
    "prevention": [
        "Maintain proper irrigation.",
        "Use fertilizers properly.",
        "Monitor regularly.",
        "Ensure sunlight.",
        "Keep area clean."
    ]
}
}

# ===============================
# HEADER
# ===============================
st.markdown('<div class="title"> Plant Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a leaf image to detect disease</div>', unsafe_allow_html=True)

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📷 Upload leaf image", type=["jpg","png","jpeg"])

# ===============================
# PREDICTION
# ===============================
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (224,224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict
    pred = model.predict(img_resized)
    class_id = np.argmax(pred)
    confidence = float(np.max(pred))

    label = class_names[class_id].strip()

    # ✅ HANDLE BOTH __ and ___
    if "___" in label:
        plant, disease = label.split("___")
    elif "__" in label:
        plant, disease = label.split("__")
    else:
        plant = label
        disease = "Unknown"

    # ✅ FETCH INFO
    info = disease_info.get(label, {
        "cause": "Not available",
        "prevention": "Consult expert"
    })

    # ===============================
    # RESULT UI
    # ===============================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(f"###  Plant: {plant}")
    st.markdown(f"###  Disease: {disease.replace('_',' ').title()}")

    with st.expander("See Causes and Preventive Measures"):
        st.write("**Cause:**", info["cause"])
        st.write("**Prevention:**", info["prevention"])

    st.markdown('</div>', unsafe_allow_html=True)