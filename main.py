from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('mobilenetv2.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file)
        image = image.resize((224, 224))  # Adjust size as per model input
        image = np.array(image) / 255.0  # Normalize if required
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
