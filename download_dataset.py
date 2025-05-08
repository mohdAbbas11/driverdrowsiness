import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import dlib
import math

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def prepare_dataset():
    """Download and prepare the dataset"""
    # Create necessary directories
    os.makedirs('dataset/train/alert', exist_ok=True)
    os.makedirs('dataset/train/drowsy', exist_ok=True)
    os.makedirs('dataset/train/yawning', exist_ok=True)
    os.makedirs('dataset/train/head_nod', exist_ok=True)
    os.makedirs('dataset/validation/alert', exist_ok=True)
    os.makedirs('dataset/validation/drowsy', exist_ok=True)
    os.makedirs('dataset/validation/yawning', exist_ok=True)
    os.makedirs('dataset/validation/head_nod', exist_ok=True)
    
    # Download dlib's facial landmark predictor
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        print("Downloading facial landmark predictor...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        download_file(url, 'shape_predictor_68_face_landmarks.dat.bz2')
        
        # Extract the bz2 file
        import bz2
        with bz2.open('shape_predictor_68_face_landmarks.dat.bz2', 'rb') as source, \
             open('shape_predictor_68_face_landmarks.dat', 'wb') as dest:
            dest.write(source.read())
        
        # Clean up
        os.remove('shape_predictor_68_face_landmarks.dat.bz2')
    
    # Initialize face detector and landmark predictor
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generate_synthetic_data(face_detector, predictor)
    
    print("Dataset preparation complete!")

def generate_synthetic_data(face_detector, predictor):
    """Generate synthetic data for training"""
    # Parameters
    num_images = 1000
    train_ratio = 0.8
    
    # Generate alert images (open eyes, normal mouth)
    for i in range(int(num_images * train_ratio)):
        img = generate_alert_image(face_detector, predictor)
        cv2.imwrite(f'dataset/train/alert/alert_{i:04d}.jpg', img)
    
    for i in range(int(num_images * (1 - train_ratio))):
        img = generate_alert_image(face_detector, predictor)
        cv2.imwrite(f'dataset/validation/alert/alert_{i:04d}.jpg', img)
    
    # Generate drowsy images (closed eyes)
    for i in range(int(num_images * train_ratio)):
        img = generate_drowsy_image(face_detector, predictor)
        cv2.imwrite(f'dataset/train/drowsy/drowsy_{i:04d}.jpg', img)
    
    for i in range(int(num_images * (1 - train_ratio))):
        img = generate_drowsy_image(face_detector, predictor)
        cv2.imwrite(f'dataset/validation/drowsy/drowsy_{i:04d}.jpg', img)
    
    # Generate yawning images (open mouth)
    for i in range(int(num_images * train_ratio)):
        img = generate_yawning_image(face_detector, predictor)
        cv2.imwrite(f'dataset/train/yawning/yawning_{i:04d}.jpg', img)
    
    for i in range(int(num_images * (1 - train_ratio))):
        img = generate_yawning_image(face_detector, predictor)
        cv2.imwrite(f'dataset/validation/yawning/yawning_{i:04d}.jpg', img)
    
    # Generate head nodding images
    for i in range(int(num_images * train_ratio)):
        img = generate_head_nod_image(face_detector, predictor)
        cv2.imwrite(f'dataset/train/head_nod/head_nod_{i:04d}.jpg', img)
    
    for i in range(int(num_images * (1 - train_ratio))):
        img = generate_head_nod_image(face_detector, predictor)
        cv2.imwrite(f'dataset/validation/head_nod/head_nod_{i:04d}.jpg', img)

def draw_eye(img, center, size, is_open=True):
    """Draw a realistic eye"""
    if is_open:
        # Draw eye socket
        cv2.ellipse(img, center, (size[0], size[1]), 0, 0, 360, (200, 200, 200), -1)
        # Draw iris
        cv2.ellipse(img, center, (size[0]//2, size[1]//2), 0, 0, 360, (100, 100, 100), -1)
        # Draw pupil
        cv2.ellipse(img, center, (size[0]//4, size[1]//4), 0, 0, 360, (0, 0, 0), -1)
        # Draw highlight
        cv2.ellipse(img, (center[0]-size[0]//4, center[1]-size[1]//4), 
                   (size[0]//8, size[1]//8), 0, 0, 360, (255, 255, 255), -1)
    else:
        # Draw closed eye
        cv2.line(img, 
                (center[0]-size[0], center[1]), 
                (center[0]+size[0], center[1]), 
                (0, 0, 0), 2)
        # Add eyelid shadow
        cv2.ellipse(img, center, (size[0], size[1]//4), 0, 0, 180, (100, 100, 100), -1)

def draw_mouth(img, center, size, is_yawning=False):
    """Draw a realistic mouth"""
    if is_yawning:
        # Draw open mouth
        cv2.ellipse(img, center, (size[0], size[1]), 0, 0, 360, (0, 0, 0), -1)
        # Draw teeth
        for i in range(-size[0]//2, size[0]//2, size[0]//8):
            cv2.line(img, 
                    (center[0]+i, center[1]-size[1]//4),
                    (center[0]+i, center[1]+size[1]//4),
                    (255, 255, 255), 1)
    else:
        # Draw normal mouth
        cv2.ellipse(img, center, (size[0], size[1]//2), 0, 0, 180, (0, 0, 0), 2)

def generate_alert_image(face_detector, predictor):
    """Generate a synthetic alert face image"""
    # Create blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw face
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (255, 255, 255), -1)
    
    # Draw realistic eyes
    draw_eye(img, (270, 200), (20, 10), is_open=True)
    draw_eye(img, (370, 200), (20, 10), is_open=True)
    
    # Draw normal mouth
    draw_mouth(img, (320, 280), (40, 20), is_yawning=False)
    
    # Add realistic skin texture
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Add subtle shadows
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (0, 0, 0), 1)
    
    return img

def generate_drowsy_image(face_detector, predictor):
    """Generate a synthetic drowsy face image"""
    # Create blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw face
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (255, 255, 255), -1)
    
    # Draw realistic closed eyes
    draw_eye(img, (270, 200), (20, 10), is_open=False)
    draw_eye(img, (370, 200), (20, 10), is_open=False)
    
    # Draw normal mouth
    draw_mouth(img, (320, 280), (40, 20), is_yawning=False)
    
    # Add realistic skin texture
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Add subtle shadows
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (0, 0, 0), 1)
    
    return img

def generate_yawning_image(face_detector, predictor):
    """Generate a synthetic yawning face image"""
    # Create blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw face
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (255, 255, 255), -1)
    
    # Draw realistic eyes
    draw_eye(img, (270, 200), (20, 10), is_open=True)
    draw_eye(img, (370, 200), (20, 10), is_open=True)
    
    # Draw yawning mouth
    draw_mouth(img, (320, 280), (40, 40), is_yawning=True)
    
    # Add realistic skin texture
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Add subtle shadows
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (0, 0, 0), 1)
    
    return img

def generate_head_nod_image(face_detector, predictor):
    """Generate a synthetic head nodding image"""
    # Create blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Random head angle between -30 and 30 degrees
    angle = np.random.uniform(-30, 30)
    
    # Create rotation matrix
    center = (320, 240)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Draw face
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (255, 255, 255), -1)
    
    # Draw realistic eyes
    draw_eye(img, (270, 200), (20, 10), is_open=True)
    draw_eye(img, (370, 200), (20, 10), is_open=True)
    
    # Draw normal mouth
    draw_mouth(img, (320, 280), (40, 20), is_yawning=False)
    
    # Add realistic skin texture
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Add subtle shadows
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (0, 0, 0), 1)
    
    # Apply rotation
    img = cv2.warpAffine(img, M, (640, 480), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return img

if __name__ == '__main__':
    prepare_dataset() 