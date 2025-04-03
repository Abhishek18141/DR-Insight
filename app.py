import os
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from preprocessing_Hybrid import preprocess_hybrid
import joblib
from PIL import Image
import requests

# assigning none to variables
binary_hybrid = None
multiclass_hybrid = None

# Load models
def load_binary_hybrid_model():
    global binary_hybrid
    if binary_hybrid is None:
        binary_hybrid = joblib.load('models/MLP_hybrid_DL2_BC.pkl')
    return binary_hybrid

def load_multiclass_hybrid_model():
    global multiclass_hybrid
    if multiclass_hybrid is None:
        multiclass_hybrid = joblib.load('models/updated_SVM_hybrid_DL2_MC.pkl')
    return multiclass_hybrid

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('HomePage.html')

# Route for Binary Classification page
@app.route('/Binary_Classification', methods=['GET', 'POST'])
def binary_classification():
    return render_template('Binay_Classification.html')

# Route for Multiclass Classification page
@app.route('/Multiclass_Classification', methods=['GET', 'POST'])
def multiclass_classification():
    return render_template('Multiclass_Classification.html')

# Route for About page
@app.route('/About_Project')
def about():
    return render_template('About_Project.html')

# Route for Contact Us page (GET request renders the contact form)
@app.route('/Contact_Us', methods=['GET'])
def contact_page():
    return render_template('Contact_Us.html')

# Formspree endpoint (replace with your Formspree URL)
FORMSPREE_URL = "https://formspree.io/f/mgvavrqz"  # Replace with your own Formspree endpoint

# Route for Contact Us page
@app.route('/Contact_Us', methods=['POST'])
def contact():
    try:
        # Retrieve form data from the request
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        # Validate data (can be enhanced further)
        if not name or not email or not subject or not message:
            return jsonify({'result': 'All fields are required.'}), 400
        
        # Prepare data to be sent to Formspree
        form_data = {
            'name': name,
            'email': email,
            'subject': subject,
            'message': message
        }

        # Send the form data to Formspree using a POST request
        response = requests.post(FORMSPREE_URL, data=form_data)

        # Check the response status
        if response.status_code == 200:
            return jsonify({'result': 'Your message has been sent successfully!'}), 200
        else:
            return jsonify({'result': 'There was an issue sending your message. Please try again later.'}), 500
    except Exception as e:
        return jsonify({'result': f'An error occurred: {str(e)}'}), 500

@app.route('/predict_binary/<model>', methods=['POST'])
def predict_binary(model):
    try:
        # Debug: Print request files
        print("Request files:", request.files)

        # Check if an image is uploaded
        if 'image' not in request.files:
            return jsonify({'result': 'No image uploaded!'}), 400

        image_file = request.files['image']

        # Debug: Print image file details
        print("Image file received:", image_file)
        print("Image filename:", image_file.filename)

        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({'result': 'No image selected!'}), 400

        # Read the image file
        image = Image.open(image_file.stream)

        # Debug: Print image details
        print("Image format:", image.format)
        print("Image size:", image.size)


        # Preprocess the image based on the selected model
        if model == 'hybrid':
            model_instance = load_binary_hybrid_model()
            processed_image = preprocess_hybrid(image)
            prediction = model_instance.predict(processed_image)
            predicted_label = "No Diabetic Retinopathy" if prediction < 0.5 else "Diabetic Retinopathy"
        else:
            return jsonify({'result': 'Invalid model type!'}), 400

        return jsonify({'result': f'{model.upper()} Model Prediction: {predicted_label}'})
    except Exception as e:
        print(f"Error during prediction: {e}")  # For debugging purposes
        return jsonify({'result': f'An error occurred: {str(e)}'}), 500

# Predict Multiclass Classification route
@app.route('/predict_multiclass/<model>', methods=['POST'])
def predict_multiclass(model):
    try:
        # Debug: Print request files
        print("Request files:", request.files)

        # Check if an image is uploaded
        if 'image' not in request.files:
            return jsonify({'result': 'No image uploaded!'}), 400

        image_file = request.files['image']

        # Debug: Print image file details
        print("Image file received:", image_file)
        print("Image filename:", image_file.filename)

        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({'result': 'No image selected!'}), 400

        # Read the image file
        image = Image.open(image_file.stream)

        # Debug: Print image details
        print("Image format:", image.format)
        print("Image size:", image.size)

        # Preprocess the image based on the selected model
        if model == 'hybrid':
            model_instance = load_multiclass_hybrid_model()
            processed_image = preprocess_hybrid(image)
            prediction = model_instance.predict(processed_image)[0]
        else:
            return jsonify({'result': 'Invalid model type!'}), 400

        # Map prediction to class label
        labels = ["No DR", "Mild Non-Proliferative DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        predicted_label = labels[int(prediction)]

        return jsonify({'result': f'{model.upper()} Model Prediction: {predicted_label}'})
    except Exception as e:
        print(f"Error during multiclass prediction: {e}")  # For debugging purposes
        return jsonify({'result': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
