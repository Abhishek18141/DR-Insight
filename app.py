import os
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from preprocessing_Hybrid import preprocess_hybrid
import joblib
import tensorflow as tf
from PIL import Image

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
        multiclass_hybrid = joblib.load('models/RF_hybrid_DL2_MC.pkl')
    return multiclass_hybrid

app = Flask(__name__)

# Update with Render PostgreSQL database URL
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://retina_database_user:79ee5LOq1e1vLveMrbbte4vtnugMaHdV@dpg-cuv1bopu0jms73a00krg-a/retina_database')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Contact us model
class ContactUs(db.Model):
    SrNo = db.Column(db.Integer, primary_key=True)
    Full_Name = db.Column(db.String(50), nullable=False)
    Email = db.Column(db.String(25), nullable=False)
    Subject = db.Column(db.String(50), nullable=False)
    Message = db.Column(db.String(250), nullable=False)
    Date_and_Time = db.Column(db.DateTime, nullable=True)

# Initialize the database
with app.app_context():
    db.drop_all()
    db.create_all()
    
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

# Route for Contact Us page
@app.route('/Contact_Us', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        Full_Name = request.form.get('name')
        Email = request.form.get('email')
        Subject = request.form.get('subject')
        Message = request.form.get('message')
        
        if Full_Name and Email and Subject and Message:
            entry = ContactUs(Full_Name=Full_Name, Email=Email, Subject=Subject, Message=Message, Date_and_Time=datetime.now())
            db.session.add(entry)
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"Error: {e}")
                return f"There was an issue saving your data to the database. Error: {str(e)}"
            
            return render_template('Contact_Us.html', message="Thank you for your message!")
        else:
            return "All fields are required."
    
    return render_template('Contact_Us.html')

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
