import cv2
import numpy as np
import serial
import time
import os
import argparse
from model.drowsiness_detector import DrowsinessDetector

class DrowsinessMonitor:
    def __init__(self, model_path=None, arduino_port=None, arduino_baudrate=9600, drowsiness_threshold=0.5, duration=3):
        """
        Initialize the drowsiness monitor
        
        Args:
            model_path (str): Path to the trained model file
            arduino_port (str): Serial port for Arduino communication
            arduino_baudrate (int): Baud rate for Arduino communication
            drowsiness_threshold (float): Threshold for drowsiness detection (0-1)
            duration (int): Duration in seconds to maintain drowsy state before alert
        """
        self.detector = DrowsinessDetector(model_path=model_path)
        self.drowsiness_threshold = drowsiness_threshold
        self.duration = duration
        self.last_alert_time = None
        self.is_alert = False
        self.arduino = None
        self.consecutive_alerts = 0
        self.required_consecutive_alerts = 3
        
        # Initialize alert states
        self.alert_states = {
            'drowsy': {'count': 0, 'threshold': 3},
            'yawning': {'count': 0, 'threshold': 3},
            'head_nod': {'count': 0, 'threshold': 3}
        }
        
        # Initialize history
        self.ear_history = []
        self.mar_history = []
        self.head_angle_history = []
        
        # Initialize Arduino if port is provided
        if arduino_port:
            try:
                self.arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino to reset
                print(f"Connected to Arduino on {arduino_port}")
            except Exception as e:
                print(f"Warning: Could not connect to Arduino: {e}")
    
    def send_arduino_command(self, command):
        """Send command to Arduino"""
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(f"{command}\n".encode())
            except Exception as e:
                print(f"Failed to send command to Arduino: {e}")
    
    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect drowsiness, yawning, and head nodding
        detection_results = self.detector.detect_drowsiness(gray)
        
        # Update alert states
        current_time = time.time()
        any_alert = False
        
        # Check each detection type
        for alert_type in ['drowsy', 'yawning', 'head_nod']:
            score = detection_results.get(alert_type + '_score', 0.0)
            if score > self.drowsiness_threshold:
                self.alert_states[alert_type]['count'] += 1
                if self.alert_states[alert_type]['count'] >= self.alert_states[alert_type]['threshold']:
                    print(f"ALERT: {alert_type.capitalize()} detected!")
                    any_alert = True
            else:
                self.alert_states[alert_type]['count'] = 0
        
        # Update overall alert state
        if any_alert:
            self.consecutive_alerts += 1
            if self.consecutive_alerts >= self.required_consecutive_alerts:
                if not self.is_alert:
                    self.last_alert_time = current_time
                    self.is_alert = True
                    self.send_arduino_command("ALERT")
                    print("ALERT: Multiple signs of drowsiness detected!")
        else:
            self.consecutive_alerts = 0
            if self.is_alert:
                self.is_alert = False
                self.send_arduino_command("AWAKE")
                print("Status: Driver is AWAKE")
        
        # Check if alert state has persisted for the required duration
        if self.is_alert and (current_time - self.last_alert_time) >= self.duration:
            return True, detection_results
        
        return False, detection_results
    
    def run(self):
        """Run the drowsiness monitoring system"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting drowsiness monitoring...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                alert, results = self.process_frame(frame)
                
                # Display status
                status = "ALERT" if alert else "AWAKE"
                color = (0, 0, 255) if alert else (0, 255, 0)  # Red for alert, green for awake
                
                # Add status text
                cv2.putText(frame, f"Status: {status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Add detection scores
                y_offset = 70
                for alert_type in ['drowsy', 'yawning', 'head_nod']:
                    score = results.get(alert_type + '_score', 0.0)
                    cv2.putText(frame, f"{alert_type.capitalize()}: {score:.2f}", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                              (255, 255, 255), 2)
                    y_offset += 40
                
                # Add consecutive alerts counter
                cv2.putText(frame, f"Consecutive Alerts: {self.consecutive_alerts}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Drowsiness Monitor', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            if self.arduino:
                self.arduino.close()

def main():
    parser = argparse.ArgumentParser(description='Drowsiness Detection System')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--arduino_port', type=str, help='Arduino serial port (e.g., COM3)')
    parser.add_argument('--baudrate', type=int, default=9600,
                       help='Baud rate for Arduino communication')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (0-1)')
    parser.add_argument('--duration', type=int, default=3,
                       help='Duration in seconds to maintain alert state')
    
    args = parser.parse_args()
    
    monitor = DrowsinessMonitor(
        model_path=args.model_path,
        arduino_port=args.arduino_port,
        arduino_baudrate=args.baudrate,
        drowsiness_threshold=args.threshold,
        duration=args.duration
    )
    monitor.run()

if __name__ == '__main__':
    main() 