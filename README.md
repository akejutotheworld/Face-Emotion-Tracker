# Emotion Detection Web Application

A comprehensive web application that detects human emotions from images or live camera feed using deep learning.

## Features

- **Image Upload**: Upload images to detect emotions
- **Live Camera**: Real-time emotion detection from webcam
- **7 Emotions**: Detects Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Database Storage**: Stores all predictions with user names and timestamps
- **Statistics Dashboard**: View emotion distribution and total predictions
- **Face Detection**: Automatic face detection using OpenCV Haar Cascades

## Project Structure

```
STUDENTS_SURNAME_MAT.matricnumber/
├── app.py                      # Flask backend server
├── model.py                    # Model training script
├── model.h5                    # Trained emotion detection model
├── emotion_database.db         # SQLite database (auto-created)
├── templates/
│   └── index.html             # Web interface
├── static/
│   └── styles.css             # CSS styling
├── requirements.txt           # Python dependencies
├── link_to_my_web_app.txt    # Deployment link
└── README.md                  # This file
```

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (First Time Only)
```bash
python model.py
```

**IMPORTANT NOTE ABOUT MODEL DATA:**

The current `model.py` uses **synthetic random data** for demonstration purposes. This means the emotion predictions will be effectively random and not accurate.

**For a production-ready academic project**, you MUST:

1. Download the **FER2013 dataset** from Kaggle:
   - Visit: https://www.kaggle.com/datasets/msambare/fer2013
   - Download the dataset
   - Extract it to a `data/` folder

2. Update `model.py` to load real data:
   ```python
   # Replace the synthetic data generation with:
   train_datagen = ImageDataGenerator(rescale=1./255, ...)
   train_generator = train_datagen.flow_from_directory(
       'data/train',
       target_size=(48, 48),
       color_mode='grayscale',
       class_mode='categorical'
   )
   # Then train with real data:
   model.fit(train_generator, epochs=50, ...)
   ```

3. Retrain the model with real data to get accurate predictions

### Step 3: Run the Application
```bash
python app.py
```

Access the app at: http://localhost:5000

## Usage

### Upload Mode
1. Enter your name
2. Click "Upload Image" tab
3. Click to upload or drag & drop an image
4. Click "Analyze Emotion"
5. View the detected emotion and confidence scores

### Live Camera Mode
1. Enter your name
2. Click "Live Camera" tab
3. Click "Start Camera" (grant camera permissions)
4. Click "Capture & Analyze" to detect emotion
5. View results

### View Statistics
- Click "View Statistics" button to see:
  - Total number of predictions
  - Distribution of detected emotions

## Database Schema

The SQLite database (`emotion_database.db`) contains:

```sql
emotion_predictions (
    id INTEGER PRIMARY KEY,
    user_name TEXT,
    timestamp TEXT,
    image_data BLOB,
    predicted_emotion TEXT,
    confidence REAL,
    capture_type TEXT
)
```

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV (face detection)
- **Image Processing**: Pillow
- **Database**: SQLite3
- **Frontend**: HTML5, CSS3, JavaScript

## Model Architecture

- Convolutional Neural Network (CNN)
- 4 Convolutional layers with Batch Normalization
- Max Pooling and Dropout for regularization
- 2 Dense layers with 512 and 256 neurons
- Softmax output layer for 7 emotion classes
- Input: 48x48 grayscale images

## Deployment

### Deploy to Replit
1. Click the "Publish" button in Replit
2. Your app will be hosted with a public URL
3. Update `link_to_my_web_app.txt` with the deployment link

### Deploy to Other Platforms
For production deployment, replace Flask's development server with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Security Notes

- The app uses Flask's development server (suitable for academic projects)
- For production: Use Gunicorn or uWSGI
- Database stores images as BLOB (consider file storage for larger scale)
- Input validation is implemented for image uploads

## Academic Requirements Met

✅ app.py - Flask backend
✅ model.py - Model training script
✅ model.h5 - Saved trained model
✅ templates/index.html - Web interface
✅ static/styles.css - Styling
✅ requirements.txt - Dependencies
✅ link_to_my_web_app.txt - Deployment link
✅ emotion_database.db - Database (auto-created)

## Known Limitations

1. **Synthetic Training Data**: The provided model is trained on random data. Replace with FER2013 for real accuracy.
2. **Single Face Detection**: Currently detects only the first face in an image.
3. **Image Size Limit**: 16MB maximum file size.
4. **Development Server**: Uses Flask's dev server (not for production scale).

## Future Improvements

- Train with real FER2013 dataset
- Add multi-face detection support
- Implement model performance metrics display
- Add data export functionality
- Create emotion history tracking per user
- Add confidence threshold filtering

## License

Academic project for educational purposes.

## Contact

Akeju Fifehanmi Titilayo
22CH031989
Fakeju.2202460@stu.cu.edu.ng
