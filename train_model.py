import os
import argparse
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def build_model(input_shape=(24, 24, 1), num_classes=4):
    """Build the CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(train_dir, val_dir, model_save_path='drowsiness_model.h5'):
    """Train the model"""
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and preprocess data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(24, 24),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(24, 24),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    # Build model
    model = build_model()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save the final model
    try:
        model.save(model_save_path)
        print(f"\nModel saved successfully at: {os.path.abspath(model_save_path)}")
    except Exception as e:
        print(f"\nError saving model: {e}")
        # Try saving in the current directory
        try:
            current_dir_model = os.path.join(os.getcwd(), 'drowsiness_model.h5')
            model.save(current_dir_model)
            print(f"Model saved in current directory: {current_dir_model}")
        except Exception as e2:
            print(f"Failed to save model in current directory: {e2}")

def main():
    parser = argparse.ArgumentParser(description='Train drowsiness detection model')
    parser.add_argument('--train_dir', type=str, required=True,
                      help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, required=True,
                      help='Directory containing validation images')
    parser.add_argument('--model_path', type=str, default='drowsiness_model.h5',
                      help='Path to save the trained model')
    
    args = parser.parse_args()
    
    print("Starting model training...")
    train_model(args.train_dir, args.val_dir, args.model_path)
    print("Training complete!")

if __name__ == '__main__':
    main() 