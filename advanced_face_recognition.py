import cv2
import numpy as np
import os
import time
from datetime import datetime
import pickle
import urllib.request
import ssl
import requests

class AdvancedFaceDetector:
    def __init__(self):
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize face recognition
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_face_images = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Create directories if they don't exist
        os.makedirs('known_faces', exist_ok=True)
        os.makedirs('face_encodings', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Initialize age detection smoothing
        self.age_history = []
        self.max_history = 5  # Keep last 5 age estimates for smoothing
    
    def load_known_faces(self):
        """Load known face images and names from files."""
        try:
            with open('face_encodings/names.pkl', 'rb') as f:
                self.known_face_names = pickle.load(f)
            print(f"Loaded {len(self.known_face_names)} known faces")
            
            # Load all face images for each person
            for name in self.known_face_names:
                # Find all images for this person
                images = [f for f in os.listdir('known_faces') if f.startswith(name)]
                if images:
                    print(f"Found {len(images)} images for {name}")
                    for image_file in images:
                        img_path = os.path.join('known_faces', image_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            self.known_face_images.append(img)
                            print(f"Loaded face image: {image_file}")
            
            # Train the recognizer if we have faces
            if self.known_face_images:
                # Create labels array - each person may have multiple images
                labels = []
                for idx, name in enumerate(self.known_face_names):
                    count = len([f for f in os.listdir('known_faces') if f.startswith(name)])
                    labels.extend([idx] * count)
                labels = np.array(labels)
                
                self.face_recognizer.train(self.known_face_images, labels)
                print(f"Face recognizer trained with {len(self.known_face_images)} images")
            else:
                print("No face images found to train the recognizer")
        except FileNotFoundError:
            print("No known faces found. Starting fresh.")
        except Exception as e:
            print(f"Error loading known faces: {str(e)}")
    
    def save_known_faces(self):
        """Save face names to file."""
        with open('face_encodings/names.pkl', 'wb') as f:
            pickle.dump(self.known_face_names, f)
    
    def detect_faces(self, frame):
        """Detect faces in the frame using OpenCV."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces, gray
    
    def download_age_model(self):
        """Download the pre-trained age detection model."""
        # Create an unverified context for HTTPS
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # URLs for the age detection model files (using a more reliable mirror)
        age_model_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_net.caffemodel"
        age_proto_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_deploy.prototxt"
        
        # Download the files using requests
        print("Downloading age detection model files...")
        
        # Download caffemodel
        print("Downloading age_net.caffemodel...")
        response = requests.get(age_model_url, verify=False)
        with open('models/age_net.caffemodel', 'wb') as f:
            f.write(response.content)
        
        # Download prototxt
        print("Downloading age_deploy.prototxt...")
        response = requests.get(age_proto_url, verify=False)
        with open('models/age_deploy.prototxt', 'wb') as f:
            f.write(response.content)
        
        print("Age detection model files downloaded successfully!")
    
    def estimate_age(self, face_img):
        """Estimate age using optimized computer vision techniques."""
        # Resize image for faster processing while maintaining aspect ratio
        max_size = 200
        height, width = face_img.shape[:2]
        scale = max_size / max(height, width)
        face_img = cv2.resize(face_img, (int(width * scale), int(height * scale)))
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Enhanced edge analysis with multiple scales
        edges_1 = cv2.Canny(normalized, 50, 150)
        edges_2 = cv2.Canny(normalized, 100, 200)
        edge_density = (np.sum(edges_1) + np.sum(edges_2)) / (2 * edges_1.shape[0] * edges_1.shape[1] * 255.0)
        
        # Multi-scale texture analysis using Gabor filters
        kernels = [
            cv2.getGaborKernel((21, 21), 3.0, 0, 10.0, 0.5, 0),
            cv2.getGaborKernel((21, 21), 4.0, np.pi/4, 10.0, 0.5, 0)
        ]
        texture_scores = []
        for kernel in kernels:
            filtered = cv2.filter2D(normalized, cv2.CV_64F, kernel)
            texture_scores.append(np.std(filtered))
        texture_score = np.mean(texture_scores)
        
        # Enhanced skin smoothness analysis
        local_var = cv2.blur(cv2.multiply(normalized, normalized), (5, 5)) - cv2.multiply(cv2.blur(normalized, (5, 5)), cv2.blur(normalized, (5, 5)))
        smoothness_score = np.mean(local_var)
        
        # Calculate face proportions
        aspect_ratio = width / height
        
        # Calculate wrinkle detection
        laplacian = cv2.Laplacian(normalized, cv2.CV_64F)
        wrinkle_score = np.mean(np.abs(laplacian))
        
        # Calculate confidence and determine age range
        age_range = ""
        
        # Refined age estimation based on multiple features with stricter thresholds
        if edge_density < 0.03 and texture_score < 600 and smoothness_score < 2 and wrinkle_score < 80:
            age_range = "0-12"
        elif edge_density < 0.05 and texture_score < 800 and smoothness_score < 3 and wrinkle_score < 120:
            age_range = "13-19"
        elif edge_density < 0.07 and texture_score < 1000 and smoothness_score < 4 and wrinkle_score < 160:
            age_range = "20-25"
        elif edge_density < 0.09 and texture_score < 1200 and smoothness_score < 5 and wrinkle_score < 200:
            age_range = "26-35"
        elif edge_density < 0.11 and texture_score < 1400 and smoothness_score < 6 and wrinkle_score < 240:
            age_range = "36-45"
        elif edge_density < 0.13 and texture_score < 1600 and smoothness_score < 7 and wrinkle_score < 280:
            age_range = "46-55"
        else:
            age_range = "55+"
        
        # Add current age estimate to history
        self.age_history.append(age_range)
        if len(self.age_history) > self.max_history:
            self.age_history.pop(0)
        
        # Use majority voting from history for more stable results
        if len(self.age_history) >= 3:
            from collections import Counter
            age_range = Counter(self.age_history).most_common(1)[0][0]
        
        # Only print debug info every 30 frames
        if hasattr(self, 'frame_counter'):
            self.frame_counter += 1
            if self.frame_counter % 30 == 0:
                print(f"Age: {age_range}")
                print(f"Edge Density: {edge_density:.3f}, Texture: {texture_score:.1f}, Smoothness: {smoothness_score:.1f}, Wrinkles: {wrinkle_score:.1f}")
        else:
            self.frame_counter = 0
        
        return age_range
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition."""
        # Resize frame for faster processing
        max_size = 800
        height, width = frame.shape[:2]
        scale = max_size / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        # Detect faces using OpenCV
        faces, gray = self.detect_faces(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # Try to recognize the face
            name = "Unknown"
            if self.known_face_images:
                label, confidence = self.face_recognizer.predict(face_gray)
                if confidence < 30:  # Even stricter threshold
                    name = self.known_face_names[label]
            
            # Estimate age
            age_range = self.estimate_age(face_img)
            
            # Draw rectangle and labels
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age_range}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def register_new_face(self, frame, name):
        """Register a new face with the given name."""
        # Detect faces
        faces, gray = self.detect_faces(frame)
        
        if len(faces) == 0:
            return False, "No face detected in the frame"
        
        # Get the first face
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        
        # Save face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"known_faces/{name}_{timestamp}.jpg"
        cv2.imwrite(filename, face_img)
        
        # Add to known faces
        self.known_face_images.append(face_gray)
        self.known_face_names.append(name)
        
        # Retrain the recognizer
        labels = np.arange(len(self.known_face_names))
        self.face_recognizer.train(self.known_face_images, labels)
        
        # Save updated names
        self.save_known_faces()
        
        return True, f"Successfully saved face image as {filename}"

def print_controls():
    """Print the available controls."""
    print("\nControls:")
    print("Press 'q' to quit")
    print("Press 'r' to register a new face")
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
                
                # Set camera properties for better performance
                print("Configuring camera settings...")
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                video_capture.set(cv2.CAP_PROP_FPS, 30)
                
                # Wait for camera to initialize
                print("Waiting for camera to initialize...")
                time.sleep(1)  # Reduced wait time
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
    face_detector = AdvancedFaceDetector()
    
    print("Initializing camera (this may take a few seconds)...")
    video_capture = init_camera()
    
    if video_capture is None:
        print("\nError: Could not initialize camera")
        print("Please make sure:")
        print("1. Your camera is properly connected")
        print("2. Camera permissions are granted in System Preferences > Security & Privacy > Privacy > Camera")
        print("3. No other application is using the camera")
        return

    print("\nAdvanced Face Recognition System Started")
    print_controls()

    frame_counter = 0
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            # Skip first few frames to allow camera to stabilize
            frame_counter += 1
            if frame_counter < 5:  # Reduced initial frame skip
                continue
            
            if not ret or frame is None:
                print("Error: Could not read frame")
                break

            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)

            # Process the frame
            processed_frame = face_detector.process_frame(frame)

            # Display the frame
            cv2.imshow('Advanced Face Recognition System', processed_frame)

            # Handle keyboard input (wait for 1ms instead of 30ms)
            key = cv2.waitKey(1) & 0xFF

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
        print("\nAdvanced Face Recognition System stopped.")

if __name__ == "__main__":
    main() 