import streamlit as st
import requests

st.title("Image Classification with Flask Backend")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:5000/predict", files={"file": uploaded_file})

            if response.status_code == 200:
                result = response.json().get("prediction", "Error: No prediction")
                st.success(f"Prediction: {result}")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the backend: {str(e)}")
