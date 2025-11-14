import os
import sqlite3
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DATABASE_NAME = 'emotion_database.db'
MODEL_PATH = 'model.h5'

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

model = None

def init_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            matric_number TEXT,
            timestamp TEXT NOT NULL,
            image_data BLOB,
            predicted_emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            capture_type TEXT NOT NULL
        )
    ''')
    
    try:
        cursor.execute('SELECT matric_number FROM emotion_predictions LIMIT 1')
    except sqlite3.OperationalError:
        cursor.execute('ALTER TABLE emotion_predictions ADD COLUMN matric_number TEXT')
    
    conn.commit()
    conn.close()
    print(f"✓ Database '{DATABASE_NAME}' initialized successfully")

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
    else:
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        print("Please run 'python model.py' first to train and save the model.")
        return False
    return True

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        resized = cv2.resize(gray, (48, 48))
        return resized.reshape(1, 48, 48, 1) / 255.0, None
    
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    resized = cv2.resize(face_roi, (48, 48))
    
    return resized.reshape(1, 48, 48, 1) / 255.0, (x, y, w, h)

def save_to_database(user_name, matric_number, image_data, emotion, confidence, capture_type):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO emotion_predictions (user_name, matric_number, timestamp, image_data, predicted_emotion, confidence, capture_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_name, matric_number, timestamp, image_data, emotion, confidence, capture_type))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.get_json()
        user_name = data.get('name', 'Anonymous')
        matric_number = data.get('matric_number', 'N/A')
        image_data = data.get('image')
        capture_type = data.get('type', 'upload')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        processed_image, face_coords = preprocess_image(image)
        
        prediction = model.predict(processed_image, verbose=0)
        emotion_idx = np.argmax(prediction[0])
        emotion = emotion_labels[emotion_idx]
        confidence = float(prediction[0][emotion_idx]) * 100
        
        save_to_database(user_name, matric_number, image_bytes, emotion, confidence, capture_type)
        
        all_emotions = {emotion_labels[i]: float(prediction[0][i]) * 100 for i in range(len(emotion_labels))}
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'all_emotions': all_emotions,
            'face_detected': face_coords is not None
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

init_database()
load_model()

if __name__ == '__main__':
    print("="*60)
    print("EMOTION DETECTION WEB APPLICATION")
    print("="*60)
    print("\nStarting Flask server...")
    print("Access the app at: http://0.0.0.0:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)
