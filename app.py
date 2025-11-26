import os
import time
# CRITICAL CHANGE: added 'render_template'
from flask import Flask, request, jsonify, render_template 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
app = Flask(__name__)
MODEL_FILE = 'multi_disease_xray_model_5class.h5' 
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASSES = ['bone_fracture', 'lung_cancer', 'normal', 'pneumonia', 'tb']
NUM_CLASSES = len(CLASSES)
# --- END CONFIGURATION ---



try:
    print(f"Loading model: {MODEL_FILE}...")
    model = load_model(MODEL_FILE, compile=False) 
    print("✅ Model loaded successfully!")
except Exception as e:
    # ... (Error handling code same as before)
    print(f"❌ ERROR: Could not load the model file '{MODEL_FILE}'.")
    print("Please ensure the model file is in the same directory as app.py.")
    print(f"Error details: {e}")
    model = None 
    

@app.route('/')
def main():
   
    return render_template('main.html') #


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
  
    if model is None:
        return jsonify({'error': 'Machine learning model not loaded. Check server logs.'}), 500

    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image selected for prediction'}), 400

    if file:
        try:
           
            img = Image.open(file.stream).convert('L') 
            img = img.resize((IMG_HEIGHT, IMG_WIDTH))
            
            # NumPy array 
            img_array = image.img_to_array(img) / 255.0 # Rescale 0-1
            
            
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            
           
            predictions = model.predict(img_array)
            
           
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_index]
            confidence = predictions[0][predicted_index] * 100
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            response = {
                'prediction': predicted_class.replace('_', ' ').title(), 
                'confidence': f'{confidence:.2f}%',
                'time': f'{elapsed_time:.3f} seconds'
            }
            return jsonify(response)

        except Exception as e:
            
            print(f"Prediction processing error: {e}")
            return jsonify({'error': f'Prediction processing failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)