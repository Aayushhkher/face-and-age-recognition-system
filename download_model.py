import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def create_age_model():
    """Create a simple CNN model for age detection."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')  # 8 age ranges
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Creating age detection model...")
    model = create_age_model()
    
    print("Saving model...")
    model.save('age_model.h5')
    print("Model saved as 'age_model.h5'")
    
    print("\nNote: This is a basic model structure. For better age detection accuracy,")
    print("you would need to train it on a large dataset of face images with age labels.")
    print("You can use datasets like IMDB-WIKI or UTKFace for training.")

if __name__ == "__main__":
    main() 