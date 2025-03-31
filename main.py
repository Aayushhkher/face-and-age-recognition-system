import cv2
import numpy as np
import os
import time
from datetime import datetime

class SimpleFaceDetector:
    def __init__(self):
        # Load OpenCV's pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces_dir = 'known_faces'
        os.makedirs(self.known_faces_dir, exist_ok=True)

    def detect_faces(self, frame):
        """Detect faces in the frame using OpenCV's cascade classifier."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def process_frame(self, frame):
        """Process a single frame for face detection."""
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(frame, 'Face Detected', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    def register_new_face(self, frame, name):
        """Save a detected face image."""
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return False, "No face detected in the frame"
        
        # Save the first detected face
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(self.known_faces_dir, filename)
        
        # Save the face image
        cv2.imwrite(filepath, face_img)
        return True, f"Successfully saved face image as {filename}"

def print_controls():
    """Print available controls to the console."""
    print("\nControls:")
    print("Press 'q' to quit")
    print("Press 'r' to register a new face")
    print("Press 'c' to clear the current frame")
    print("Press 'h' to show this help message\n")

def init_camera():
    """Initialize camera with proper settings for macOS."""
    print("Attempting to initialize camera...")
    
    # Try different camera indices that might work on Mac
    camera_indices = [0, 1]
    
    for idx in camera_indices:
        print(f"Trying camera index {idx}...")
        video_capture = cv2.VideoCapture(idx)
        
        if video_capture.isOpened():
            # Try to read multiple test frames
            success = True
            for _ in range(5):  # Try reading 5 frames
                ret, frame = video_capture.read()
                if not ret or frame is None:
                    success = False
                    break
                time.sleep(0.1)  # Small delay between reads
            
            if success:
                print(f"Successfully connected to camera {idx}")
                
                # Set camera properties
                print("Configuring camera settings...")
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                video_capture.set(cv2.CAP_PROP_FPS, 30)
                
                # Wait for camera to initialize
                print("Waiting for camera to initialize...")
                time.sleep(2)
                return video_capture
            else:
                print(f"Camera {idx} opened but couldn't read frames consistently")
                video_capture.release()
        else:
            print(f"Failed to open camera {idx}")
    
    print("Could not initialize any camera")
    return None

def main():
    # Initialize face detector
    face_detector = SimpleFaceDetector()
    
    print("Initializing camera (this may take a few seconds)...")
    video_capture = init_camera()
    
    if video_capture is None:
        print("\nError: Could not initialize camera")
        print("Please make sure:")
        print("1. Your camera is properly connected")
        print("2. Camera permissions are granted in System Preferences > Security & Privacy > Privacy > Camera")
        print("3. No other application is using the camera")
        return

    print("\nFace Detection System Started")
    print_controls()

    frame_counter = 0
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            # Skip first few frames to allow camera to stabilize
            frame_counter += 1
            if frame_counter < 10:
                continue
            
            if not ret or frame is None:
                print("Error: Could not read frame")
                break

            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)

            # Process the frame
            processed_frame = face_detector.process_frame(frame)

            # Display the frame
            cv2.imshow('Face Detection System', processed_frame)

            # Handle keyboard input (wait for 30ms)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):  # Quit
                break
            elif key == ord('r'):  # Register new face
                name = input("Enter the name for the new face: ")
                success, message = face_detector.register_new_face(frame, name)
                print(message)
            elif key == ord('h'):  # Show help
                print_controls()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up
        video_capture.release()
        cv2.destroyAllWindows()
        print("\nFace Detection System stopped.")

if __name__ == "__main__":
    main() 