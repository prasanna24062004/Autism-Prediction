# app.py
import streamlit as st
from predict import predict_autism
from PIL import Image
import os

st.title("🧠 Autism Prediction from Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = os.path.join("images", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict_autism(img_path)
        st.success(result)
