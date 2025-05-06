# predict.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_autism(img_path):
    model = load_model('models/autism_model.h5')  # Adjust path if needed

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    return "Autism Detected" if prediction >= 0.5 else "No Autism Detected"
