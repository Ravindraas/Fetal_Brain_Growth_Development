import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import load_model
from PIL import Image
# Load the saved model
model = load_model("child_brain_development_model.h5")
# Function to predict on direct image input
def predict_on_direct_image(model, image, img_size=64):
    if image is None:
        st.error("Invalid image input")
        return None
    img = cv2.resize(image, (img_size, img_size))
    img = preprocess_input(np.array([img]))
    prediction = model.predict(img)
    prediction_binary = (prediction > 0.5).astype(int)
    if prediction_binary == 1:
        return "Developing"
    else:
        return "Not Developing"
# Streamlit app
st.title("Child Brain Development Prediction")
st.write("Upload an image to predict whether the brain is developing or not.")
# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # Predict and display the result
    result = predict_on_direct_image(model, img_array)
    if result is not None:
        st.write(f"The prediction for the uploaded image is: *{result}*")