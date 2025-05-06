import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
import joblib

# Load model
model = joblib.load("autism_svm_model.pkl")

# Preprocess function
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    features, _ = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      orientations=9, visualize=True)
    return np.array([features])

# Predict function
def predict(image):
    features = process_image(image)
    prediction = model.predict(features)[0]
    label = "Autistic" if prediction == 0 else "Non-Autistic"
    return label

# UI
st.title("🧠 Autism Prediction from Face")
option = st.radio("Choose Input Method:", ("📷 Use Webcam", "📁 Upload Image"))

if option == "📷 Use Webcam":
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    capture = st.button("📸 Capture Image")

    frame_captured = None
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if capture:
            frame_captured = frame
            break

    camera.release()

    if frame_captured is not None:
        st.subheader("Captured Image")
        st.image(frame_captured, channels="BGR", use_container_width=True)
        label = predict(frame_captured)
        st.success(f"Prediction: {label}")

elif option == "📁 Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded)
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        st.subheader("Uploaded Image")
        st.image(img, use_container_width=True)

        label = predict(img_bgr)
        st.success(f"Prediction: {label}")
