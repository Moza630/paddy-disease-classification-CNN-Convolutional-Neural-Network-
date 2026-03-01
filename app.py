import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
import random
from PIL import Image

# Set page config for a premium look
st.set_page_config(
    page_title="Paddy Guard - Disease Classification",
    page_icon="🌾",
    layout="wide",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .label-header {
        color: #2e7d32;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD METADATA AND MODEL ---
@st.cache_resource
def load_resources():
    try:
        with open('paddy_model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load model (handling potential path issues)
        model = tf.keras.models.load_model(metadata['model_path'])
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model or metadata: {e}")
        return None, None

model, metadata = load_resources()

# --- HELPER FUNCTIONS ---
def predict_image(img):
    if model and metadata:
        img = img.resize((metadata['img_size'], metadata['img_size']))
        img_array = np.array(img).astype('float32') # Keeping it in [0, 255] to match training
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        label = metadata['labels'][class_idx]
        
        return label, confidence
    return "Error", 0.0

def predict_text(text):
    text = text.lower()
    # Keyword-based prediction logic (Symptom Mapping)
    symptoms = {
        "bacterial_leaf_blight": ["yellowish-white", "water-soaked stripes", "wavy margins", "leaf drying"],
        "bacterial_leaf_streak": ["dark green", "water-soaked streaks", "narrow stripes", "bacterial ooze"],
        "bacterial_panicle_blight": ["rotting panicles", "grain discoloration", "empty grains"],
        "blast": ["diamond-shaped lesions", "eye-shaped spots", "neck rot", "gray centers"],
        "brown_spot": ["circular spots", "oval spots", "brown lesions", "yellow halo"],
        "dead_heart": ["central leaf drying", "stem borer", "withered heart"],
        "downy_mildew": ["stunted growth", "whitish downy growth", "deformed leaves"],
        "hispa": ["white streaks", "scraped leaves", "skeletonized leaves"],
        "tungro": ["stunted growth", "orange-yellow leaves", "discolored leaves"],
        "normal": ["green leaves", "healthy", "no spots", "vigorous growth"]
    }
    
    # Calculate scores for each class
    scores = {disease: 0 for disease in symptoms}
    found_any = False
    for disease, keywords in symptoms.items():
        for kw in keywords:
            if kw in text:
                scores[disease] += 1
                found_any = True
    
    if not found_any:
        return "Uncertain - Not enough keywords found", 0.0
    
    predicted_disease = max(scores, key=scores.get)
    # Mock confidence for text
    confidence = 0.85 if scores[predicted_disease] > 1 else 0.65
    return predicted_disease, confidence

# --- UI LAYOUT ---
st.title("🌾 Paddy Guard: Disease Classification")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Classification Mode")
    mode = st.radio("Select Prediction Type:", ["📸 Image Upload", "✍️ Disease Description (Text)"], horizontal=True)

    # Random Button Logic
    if st.button("🎲 Random Sample"):
        if mode == "📸 Image Upload":
            test_dir = "test_images"
            if os.path.exists(test_dir):
                random_img = random.choice(os.listdir(test_dir))
                st.session_state.uploaded_file = os.path.join(test_dir, random_img)
                st.session_state.random_triggered = True
        else:
            sample_texts = [
                "The leaves show diamond-shaped lesions with gray centers and reddish-brown margins.",
                "Rice panicles are turning brown and rotting, with many empty grains observed.",
                "There are narrow, dark green, water-soaked streaks on the leaf blades.",
                "Plants are stunted and the young leaves are showing orange-yellow discoloration.",
                "Circular brown spots with a yellow halo are visible across the older leaves."
            ]
            st.session_state.input_text = random.choice(sample_texts)

with col2:
    st.subheader("Analysis & Result")
    result_container = st.empty()

# --- PREDICTION FLOW ---
if mode == "📸 Image Upload":
    uploaded_file = st.file_uploader("Upload a paddy leaf image...", type=["jpg", "png", "jpeg"])
    
    # Handle Random selection
    if 'random_triggered' in st.session_state and st.session_state.random_triggered:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="Randomly Selected Image", use_column_width=True)
        del st.session_state.random_triggered
    elif uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        image = None

    if image and st.button("Predict Disease"):
        with st.spinner("Analyzing image patterns..."):
            label, confidence = predict_image(image)
            with col2:
                st.markdown(f"""
                <div class="prediction-card">
                    <p class="label-header">{label.replace('_', ' ').title()}</p>
                    <p>Confidence: <b>{confidence*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

else:
    # Text Mode
    input_text = st.text_area("Describe the symptoms (e.g., spots, discoloration, streaks):", 
                             value=st.session_state.get('input_text', ""),
                             placeholder="Enter symptoms here...",
                             height=150)
    
    if st.button("Analyze Symptoms"):
        if input_text:
            with st.spinner("Analyzing symptom keywords..."):
                label, confidence = predict_text(input_text)
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <p class="label-header">{label.replace('_', ' ').title()}</p>
                        <p>Keyword Match Confidence: <b>{confidence*100:.0f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    if "Uncertain" not in label:
                        st.info("Tip: This prediction is based on detected symptom keywords. For a more accurate diagnosis, please upload an image.")
        else:
            st.warning("Please enter some text or click 'Random Sample'.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Developed for Paddy Health Monitoring & Classification</p>", unsafe_allow_html=True)
