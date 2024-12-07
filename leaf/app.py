from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pickle
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import skimage
from getRemidies import get_remedies_from_google
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Upload folder configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Allowed file extensions for image files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process the image and make prediction

def load_image(file):
    dimension=(104, 104)
    image = Image.open(file)
    flat_data = []
    img = skimage.io.imread(file)
    img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
    flat_data.append(img_resized.flatten()) 
    return image,flat_data

def predict_disease(image_path):
    plot , img = load_image(image_path)
    k = ['Bacterial leaf blight', 'Leaf smut', 'Brown spot']
    p = model.predict(img)

    s = [str(i) for i in p] 
    a = int("".join(s)) 
    print("Predicted Disease is", k[a])
    return k[a]
        # Open image
    # img = Image.open(image_path)
    # # Resize the image to match the expected size (104x104)
    # img = img.resize((104, 104))  # Resize to match model input size
    
    # # Convert the image to a numpy array
    # img = np.array(img)
    
    # # Normalize the image (this may depend on how your model was trained, adjust if needed)
    # img = img / 255.0  # Normalize pixel values to [0, 1]
    
    # # Flatten the image into a 1D array
    # img_flattened = img.flatten().reshape(1, -1)  # Flatten and reshape to (1, num_features)
    
    # # Predict the disease using the model
    # prediction = model.predict(img_flattened)
    
    # # Assuming you have a dictionary to map classes to disease names
    # disease_names = {0: 'Healthy', 1: 'Rice Blast', 2: 'Bacterial Leaf Blight'}  # Example
    # return disease_names.get(prediction[0], 'Unknown Disease')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Call your prediction function here
        disease = predict_disease(filepath)
        remedies = get_remedies_from_google(disease)
        # Display the image and prediction result
        return render_template('result.html', disease=disease, filepath=filepath, remedies=remedies)
    else:
        return "Invalid file type. Please upload an image."

if __name__ == '__main__':
    app.run(debug=True)
