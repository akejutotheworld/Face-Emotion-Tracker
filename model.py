import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def create_emotion_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    return model

def train_model_with_sample_data():
    print("Creating emotion detection model...")
    model = create_emotion_model()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    print("\nGenerating synthetic training data for demonstration...")
    num_samples = 1000
    X_train = np.random.rand(num_samples, 48, 48, 1).astype('float32')
    y_train = keras.utils.to_categorical(np.random.randint(0, 7, num_samples), 7)
    
    X_val = np.random.rand(200, 48, 48, 1).astype('float32')
    y_val = keras.utils.to_categorical(np.random.randint(0, 7, 200), 7)
    
    print("\nTraining model (with synthetic data for demonstration)...")
    print("Note: For production, use FER2013 or similar emotion dataset")
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=5,
        verbose=1
    )
    
    print("\nSaving model as 'model.h5'...")
    model.save('model.h5')
    print("Model saved successfully!")
    
    print("\nTraining complete!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("EMOTION DETECTION MODEL TRAINING SCRIPT")
    print("="*60)
    print("\nThis script creates and trains a CNN model for emotion detection")
    print("Emotions detected: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise")
    print("\nFor academic purposes, this uses synthetic data.")
    print("For real-world use, download FER2013 dataset from Kaggle.")
    print("="*60)
    
    if not os.path.exists('model.h5'):
        model = train_model_with_sample_data()
        print("\n✓ Model training completed and saved!")
    else:
        print("\nModel file 'model.h5' already exists.")
        print("Delete it first if you want to retrain.")
