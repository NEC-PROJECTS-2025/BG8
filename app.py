from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.utils import get_custom_objects

# Initialize Flask app
app = Flask(__name__)

# Set upload folder for images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the focal loss function
def focal_loss_fixed(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + 1e-8)) - tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + 1e-8))

# Register custom loss function
get_custom_objects().update({'focal_loss_fixed': focal_loss_fixed})

# Load the trained lung cancer model
try:
    model_path = 'Model/improved_model_1 (1).h5'  # Ensure this path is correct
    model_2 = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_2 = None

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    """Render the prediction page."""
    return render_template('Prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict lung cancer from an uploaded image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Step 1: Read and preprocess the image (same as in Colab)
        img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
        img = img.resize((50, 50))  # Resize to 50x50
        image_array = np.array(img)  # Convert to numpy array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
        image_array = image_array / 255.0  # Normalize the image to [0, 1]

        # Step 2: Check if model is loaded
        if model_2 is None:
            return jsonify({'error': 'Model not loaded successfully'}), 500

        # Step 3: Make prediction
        prediction = model_2.predict(image_array)
        probabilities = prediction[0]
        
        # Step 4: Use threshold-based prediction logic (same as in Colab)
        threshold = 0.6  # Set the threshold for classification
        if np.max(probabilities) >= threshold:
            predicted_class = np.where(probabilities == np.max(probabilities))[0][0]
            confidence = np.max(probabilities) * 100  # Confidence as percentage
        else:
            predicted_class = None  # Uncertain prediction
            confidence = 0

        # Step 5: Map prediction to human-readable labels
        class_labels = ['No Cancer', 'Cancer']
        if predicted_class is not None:
            predicted_label = class_labels[predicted_class]
            response = {
    "predicted": int(predicted_class),  # Convert to standard int
    "label": predicted_label,
    "confidence": round(float(confidence), 2),  # Convert to standard float
    "probabilities": [float(prob) for prob in probabilities]  # Convert each to float
}

        else:
            response = {
                "error": "The model is uncertain about the prediction",
                "probabilities": probabilities.tolist()
            }

        print(f"🔍 Prediction: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return jsonify({'error': 'Error processing the file', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
