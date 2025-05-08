import argparse
import os
import sys
from drowsiness_monitor import DrowsinessMonitor

def parse_args():
    parser = argparse.ArgumentParser(description='Drowsiness Detection System')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (default: use drowsiness_model.h5)')
    parser.add_argument('--arduino_port', type=str, default='COM3',
                        help='Arduino serial port (default: COM3)')
    parser.add_argument('--arduino_baudrate', type=int, default=9600,
                        help='Arduino serial baudrate (default: 9600)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Drowsiness detection threshold (default: 0.7)')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Duration in seconds to consider someone drowsy (default: 2.0)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if model exists if specified
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return 1
    
    try:
        # Initialize the drowsiness monitor
        monitor = DrowsinessMonitor(
            model_path=args.model_path,
            arduino_port=args.arduino_port,
            arduino_baudrate=args.arduino_baudrate
        )
        
        # Set custom thresholds if provided
        monitor.drowsy_threshold = args.threshold
        monitor.drowsy_duration = args.duration
        
        # Run the monitor
        print(f"Starting drowsiness monitor with threshold {args.threshold} and duration {args.duration}s")
        monitor.run()
        
        return 0
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 