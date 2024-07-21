from flask import Flask, render_template, request, redirect, url_for
from flask_mail import Mail, Message 
import numpy as np
import os




from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Debugging: Print current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in the model directory:", os.listdir('model'))



# Load model
try:
    model = load_model("model/deep_model.h5")
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None

def prediction_disease(cotton_plant):
    if not model:
        return "Model loading error", 'error.html'
    
    test_image = load_img(cotton_plant, target_size=(150, 150))
    print("Got image for the prediction")
    
    test_image = img_to_array(test_image) / 255  # Converting it into np array and normalizing it
    test_image = np.expand_dims(test_image, axis=0)  # Change the dimension from 3D to 4D
    
    result = model.predict(test_image).round(3)  # It predicts whether the plant is diseased or not
    print("Raw result:", result)
    
    pred = np.argmax(result)  # Get the index of maximum value
    
    if pred == 0:
        return "Diseased Cotton Plant", 'bacterial-blight.html'  # If index is 0
    elif pred == 1:
        return "Diseased Cotton Plant", 'curl-virus.html'  # If index is 1
    elif pred == 2:
        return "Healthy Cotton Plant", 'chemical-fertilizers.html'  # If index is 2
    elif pred == 3:
        return "Healthy Cotton Plant", 'healthy_plant.html'  # If index is 3
    elif pred == 4:
        return "Healthy Cotton Plant", 'healthy_plant.html'  # If index is 4
    else:
        return "Diseased Cotton Plant", 'churda_mava_rog.html'  # If index is 5

# Create a Flask instance
app = Flask(__name__)


# Render the index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/FAQs')
def FAQ():
    return render_template('FAQ.html')

# Get input image from client, then predict class and render respective .html page
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # File input
        filename = file.filename
        
        print("Input posted:", filename)
        
        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)
        
        print("Predicting class...")
        pred, output_page = prediction_disease(cotton_plant=file_path)
        
        return render_template(output_page, pred_output=pred, user_image=file_path)



 
# For local system and cloud
if __name__ == "__main__":
    app.run(debug=True)
