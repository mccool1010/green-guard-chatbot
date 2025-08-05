import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin req33uests

# Load trained model
MODEL_PATH = r"C:/Users/asus/OneDrive/Documents/project final/okra_leaf_mobilenetv2.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Image Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Class labels (fixed index issue)
class_names = {
    0: "Alternaria Leaf Spot",
    1: "Cercospora Leaf Spot",
    2: "Downy Mildew",
    3: "Healthy",
    4: "Leaf Curly Virus",
    5: "Phyllosticta Leaf Spot",
    6: "Bhendi Yellow Vein Mosaic Disease"
}

# Function to process and predict disease from image
def predict_disease(image_path):
    try:
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Model input shape

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)  # Get highest probability class
        confidence = float(np.max(prediction))  # Get confidence score

        # Get disease name or return "Unknown"
        predicted_class = class_names.get(predicted_class_index, "Unknown Disease")
        return predicted_class, confidence

    except Exception as e:
        return f"Error processing image: {str(e)}", None

# Route to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_path = "temp.jpg"
        file.save(image_path)

        predicted_disease, confidence = predict_disease(image_path)
        os.remove(image_path)  # Clean up

        if confidence is None:
            return jsonify({"error": predicted_disease}), 500

        return jsonify({
            "predicted_disease": predicted_disease,
            "confidence": f"{confidence * 100:.2f}%"  # Convert to percentage
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)