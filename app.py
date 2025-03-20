from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import traceback
from PIL import Image
import io
from tensorflow.keras.utils import get_custom_objects
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained CT vs Non-CT classifier model
try:
    ct_classifier_path = 'Model/ct_vs_nonct_classifier.h5'
    ct_classifier = tf.keras.models.load_model(ct_classifier_path)
    print("✅ CT vs Non-CT classifier loaded successfully")
except Exception as e:
    print(f"❌ Error loading CT classifier model: {e}")
    ct_classifier = None

def focal_loss_fixed(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + 1e-8)) - tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + 1e-8))

# Register custom loss function
get_custom_objects().update({'focal_loss_fixed': focal_loss_fixed})

# Load the trained lung cancer models
try:
    model_1_path = 'Model/improved_model_1 (1).h5'  # Update with the correct path
    model_2_path = 'Model/improved_model_2.h5'  # Update with the correct path
    model_3_path = 'Model/improved_model_3_updated.h5'  # Update with the correct path

    model_1 = tf.keras.models.load_model(model_1_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
    model_2 = tf.keras.models.load_model(model_2_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
    model_3 = tf.keras.models.load_model(model_3_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})

    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_1, model_2, model_3 = None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('Prediction.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

def is_ct_scan(image):
    """
    Validate if the uploaded image is a CT scan using the trained classifier.
    """
    try:
        # Convert grayscale to 3-channel RGB
        if len(image.shape) == 2:  # If grayscale (H, W)
            image = np.stack((image,) * 3, axis=-1)  # Convert to (H, W, 3)

        # Resize and normalize image
        img_resized = cv2.resize(image, (50, 50))
        img_resized = img_resized / 255.0  # Normalize
        img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

        # Predict with the CT classifier
        y_pred = ct_classifier.predict(img_resized)
        predicted_class = int(y_pred[0][0] > 0.5)  # Assuming sigmoid output (0 = CT, 1 = Non-CT)

        print(f"🔹 CT Classifier Prediction: {y_pred[0][0]} | Classified as: {'CT' if predicted_class == 0 else 'Non-CT'}")

        return predicted_class == 0  # Return True if CT Scan

    except Exception as e:
        print(f"❌ Error in CT scan classification: {e}")
        return False  # Default to rejecting invalid images

@app.route('/predict', methods=['POST'])
def predict():
    """Predict lung cancer from an uploaded image using an ensemble of 3 models after verifying it's a CT scan."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read the image
        img = Image.open(io.BytesIO(file.read()))

        # Convert to OpenCV format (grayscale)
        image_cv = np.array(img.convert('L'))  # Convert to grayscale (H, W)

        # Validate if it's a CT scan using the trained classifier
        if not is_ct_scan(image_cv):
            return jsonify({'error': 'Invalid image type. Please upload a Lung CT Scan.'}), 400

        # Resize and preprocess the image for cancer prediction
        img_resized = img.resize((50, 50)).convert('L')
        image_array = np.array(img_resized) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=(0, -1))  # Shape (1, 50, 50, 1)

        # Predictions from each cancer detection model
        y_pred_prob_1 = model_1.predict(image_array)
        y_pred_prob_2 = model_2.predict(image_array)
        y_pred_prob_3 = model_3.predict(image_array)

        # Ensemble prediction (average probability)
        y_pred_prob_ensemble = (y_pred_prob_1 + y_pred_prob_2 + y_pred_prob_3) / 3
        predicted_class = np.argmax(y_pred_prob_ensemble[0])
        
       # predicted_class = 1 if y_pred_prob_ensemble[0][1] > threshold else 0
        confidence = y_pred_prob_ensemble[0][predicted_class] * 100
        print(predicted_class,y_pred_prob_ensemble)


        class_labels = ['No Cancer', 'Cancer']
        predicted_label = class_labels[predicted_class]

        response = {
    "predicted": int(predicted_class),  # Convert to native Python int
    "label": predicted_label,
    "confidence": round(confidence, 2),
    "probabilities": [float(prob) for prob in y_pred_prob_ensemble[0]]  # Ensure probabilities are float
}


        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Error processing the file', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
