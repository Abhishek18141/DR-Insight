import os
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from preprocessing_ML_DL_Hybrid import preprocess_dl_image, preprocess_hybrid_binary_image, preprocess_hybrid_multiclass_image
import joblib
import tensorflow as tf
from PIL import Image

# Load models with debugging statements
try:
    print("Loading Deep Learning models...")
    binary_dl = tf.keras.models.load_model('models/binary_classification_InceptionV3.h5')
    multiclass_dl = tf.keras.models.load_model('models/multiclass_model_inceptionv3.h5')
    print("Deep Learning models loaded successfully.")
except Exception as e:
    print(f"Error loading DL models: {e}")

try:
    print("Loading Hybrid models...")
    binary_hybrid = joblib.load('models/MLP_hybrid_DL2_BC.pkl')
    multiclass_hybrid = joblib.load('models/MLP_hybrid_DL1_MC.pkl')
    print("Hybrid models loaded successfully.")
except Exception as e:
    print(f"Error loading Hybrid models: {e}")

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://retina_database_user:79ee5LOq1e1vLveMrbbte4vtnugMaHdV@dpg-cuv1bopu0jms73a00krg-a/retina_database')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Contact Us Model
class ContactUs(db.Model):
    SrNo = db.Column(db.Integer, primary_key=True)
    Full_Name = db.Column(db.String(50), nullable=False)
    Email = db.Column(db.String(25), nullable=False)
    Subject = db.Column(db.String(50), nullable=False)
    Message = db.Column(db.String(250), nullable=False)
    Date_and_Time = db.Column(db.String(12), nullable=True)

@app.route('/')
def home():
    return render_template('HomePage.html')

@app.route('/predict_binary/<model>', methods=['POST'])
def predict_binary(model):
    try:
        print(f"Received request for binary prediction using {model} model.")
        if 'image' not in request.files:
            print("No image file in request.")
            return jsonify({'result': 'No image uploaded!'}), 400
        
        image_file = request.files['image']
        print(f"Received image: {image_file.filename}")
        if image_file.filename == '':
            return jsonify({'result': 'No image selected!'}), 400
        
        image = Image.open(image_file.stream)
        print(f"Image format: {image.format}, Image size: {image.size}")
        
        if model == 'dl':
            processed_image = preprocess_dl_image(image)
            print("Processed image for DL model.")
            prediction = binary_dl.predict(processed_image)
            print(f"DL Prediction: {prediction}")
            predicted_label = "No Diabetic Retinopathy" if prediction < 0.5 else "Diabetic Retinopathy"
        elif model == 'hybrid':
            processed_image = preprocess_hybrid_binary_image(image)
            print("Processed image for Hybrid model.")
            prediction = binary_hybrid.predict(processed_image)
            print(f"Hybrid Prediction: {prediction}")
            predicted_label = "No Diabetic Retinopathy" if prediction < 0.5 else "Diabetic Retinopathy"
        else:
            print("Invalid model type!")
            return jsonify({'result': 'Invalid model type!'}), 400

        return jsonify({'result': f'{model.upper()} Model Prediction: {predicted_label}'})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'result': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
