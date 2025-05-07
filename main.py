import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('mobilenetv2.keras')

# Streamlit App Layout
st.title("Malignant vs Benign Prediction")

st.markdown("""
    Upload an image of the tumor, and the model will predict whether it is benign or malignant.
""")

# File uploader widget
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    # Open and process the image
    image = Image.open(file)
    image = image.resize((224, 224))  # Adjust size as per model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)
    result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'

    # Display the result
    st.image(image[0], caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: **{result}**")
