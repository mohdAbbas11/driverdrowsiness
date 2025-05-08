import os
import tensorflow as tf
from keras import layers, models
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import dlib

class DrowsinessDetector:
    def __init__(self, model_path=None):
        """
        Initialize the drowsiness detector
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = None
        self.model_path = 'drowsiness_model.h5'
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize dlib's face detector and facial landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Thresholds and parameters
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.5  # Mouth Aspect Ratio threshold
        self.HEAD_NOD_THRESHOLD = 20.0  # Degrees
        self.CONSECUTIVE_FRAMES = 3
        
        # Initialize history
        self.ear_history = []
        self.mar_history = []
        self.head_angle_history = []
        
        if model_path and os.path.exists(model_path):
            self.model = models.load_model(model_path)
        else:
            self.model = self._build_model()
        
    def _build_model(self):
        """Build the CNN model architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(24, 24, 1)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')  # 4 classes: alert, drowsy, yawning, head_nod
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_dir, val_dir, epochs=10, batch_size=32):
        """Train the model on the provided dataset"""
        # Data augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Load and preprocess data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(24, 24),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(24, 24),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        return history
    
    def load_model(self, model_path='drowsiness_model.h5'):
        """Load the trained model"""
        try:
            self.model = models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default model architecture...")
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default model architecture if no model file is found"""
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4, activation='softmax')  # Updated to 4 classes
        ])
        print("Default model created")
    
    def calculate_ear(self, eye_landmarks):
        """Calculate the Eye Aspect Ratio (EAR)"""
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """Calculate the Mouth Aspect Ratio (MAR)"""
        # Compute the euclidean distances between the vertical mouth landmarks
        A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
        B = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])
        
        # Calculate the mouth aspect ratio
        mar = A / B
        return mar
    
    def calculate_head_angle(self, face_landmarks):
        """Calculate head angle using facial landmarks"""
        # Calculate angle between eyes and nose
        left_eye = np.mean(face_landmarks[36:42], axis=0)
        right_eye = np.mean(face_landmarks[42:48], axis=0)
        nose_tip = face_landmarks[30]
        
        eye_center = (left_eye + right_eye) / 2
        angle = np.degrees(np.arctan2(nose_tip[1] - eye_center[1], nose_tip[0] - eye_center[0]))
        
        return angle
    
    def preprocess_eye(self, eye):
        """Preprocess detected eye for the model"""
        # Resize to 24x24 pixels
        eye = cv2.resize(eye, (24, 24))
        
        # Convert to grayscale if not already
        if len(eye.shape) > 2:
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        eye = cv2.equalizeHist(eye)
        
        # Apply Gaussian blur to reduce noise
        eye = cv2.GaussianBlur(eye, (3, 3), 0)
        
        # Normalize
        eye = eye / 255.0
        
        # Add channel dimension
        eye = np.expand_dims(eye, axis=-1)
        return eye
    
    def detect_drowsiness(self, frame):
        """Detect drowsiness in a frame"""
        # Convert frame to grayscale if it's not already
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect faces
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return {
                'drowsiness_score': 0.0,
                'yawning_score': 0.0,
                'head_nod_score': 0.0,
                'state': 'no_face'
            }
        
        # Get facial landmarks
        landmarks = self.predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Calculate EAR
        left_ear = self.calculate_ear(landmarks[36:42])
        right_ear = self.calculate_ear(landmarks[42:48])
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR
        mar = self.calculate_mar(landmarks[48:68])
        
        # Calculate head angle
        head_angle = self.calculate_head_angle(landmarks)
        
        # Get model prediction
        face_roi = gray[faces[0].top():faces[0].bottom(), faces[0].left():faces[0].right()]
        face_roi = cv2.resize(face_roi, (24, 24))
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
        face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension
        
        prediction = self.model.predict(face_roi, verbose=0)[0]
        
        # Determine state based on all indicators
        drowsiness_score = prediction[1]  # Drowsy class
        yawning_score = prediction[2]  # Yawning class
        head_nod_score = prediction[3]  # Head nod class
        
        state = 'alert'
        if drowsiness_score > 0.5 or ear < self.EAR_THRESHOLD:
            state = 'drowsy'
        elif yawning_score > 0.5 or mar > self.MAR_THRESHOLD:
            state = 'yawning'
        elif head_nod_score > 0.5 or abs(head_angle) > self.HEAD_NOD_THRESHOLD:
            state = 'head_nod'
        
        return {
            'drowsiness_score': drowsiness_score,
            'yawning_score': yawning_score,
            'head_nod_score': head_nod_score,
            'state': state
        }
    
    def predict(self, image):
        """Predict on a single image
        
        Args:
            image: Grayscale image of shape (24, 24, 1)
            
        Returns:
            Class prediction (0: alert, 1: drowsy) and confidence
        """
        if self.model is None:
            if not self.load_model():
                print("No model available. Please train or load a model first.")
                return None
        
        # Ensure image is properly formatted
        if image.shape != (24, 24, 1):
            print(f"Expected image shape (24, 24, 1), got {image.shape}")
            return None
        
        # Normalize the image
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Get prediction
        prediction = self.model.predict(image)[0]
        
        # Get class with highest probability
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        return {
            'class': predicted_class,  # 0: alert, 1: drowsy
            'confidence': float(confidence),
            'prediction': prediction
        }
        
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def save_model(self, model_path):
        """Save the trained model"""
        self.model.save(model_path)


if __name__ == "__main__":
    # Example usage
    detector = DrowsinessDetector()
    detector.build_model()
    
    # To train the model:
    # history = detector.train('path/to/train_data', 'path/to/val_data')
    # detector.plot_training_history(history)
    
    # To use pre-trained model:
    # detector.load_model()
    # result = detector.predict(your_image)
    # print(result) 