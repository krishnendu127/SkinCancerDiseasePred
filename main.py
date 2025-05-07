import os
import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from groq import Groq

# Initialize the Groq client
client = Groq(api_key="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")

# Load the pre-trained model for tumor prediction
tumor_model = tf.keras.models.load_model('mobilenetv2.keras')

# Streamlit App Layout
st.title("Malignant vs Benign Prediction and Chatbot")

# Tumor Prediction Section
st.subheader("Tumor Image Prediction")
st.markdown("""
    Upload an image of the tumor, and the model will predict whether it is benign or malignant.
""")

# File uploader widget for tumor image
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    # Open and process the image
    image = Image.open(file)
    image = image.resize((224, 224))  # Adjust size as per model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)

    # Prediction for tumor
    prediction = tumor_model.predict(image)
    result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'

    # Display the result
    st.image(image[0], caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: **{result}**")

    # Automatically feed the result to the chatbot
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Provide details about a {result} tumor."}
        ],
        model="gemma2-9b-it",
    )

    st.write(f"Chatbot Response: {chat_completion.choices[0].message.content}")

# Chatbot Section
st.subheader("AI Chatbot")
st.markdown("""
    Chat with the AI chatbot powered by Groq's Gemma 2B. You can ask further questions related to the prediction result.
""")

# Text input for chatbot interaction
user_input = st.text_input("Ask me anything related to the prediction:")

if user_input:
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": user_input}
        ],
        model="gemma2-9b-it",
    )

    st.write(f"Chatbot Response: {chat_completion.choices[0].message.content}")
