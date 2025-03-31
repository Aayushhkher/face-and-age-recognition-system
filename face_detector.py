import cv2
import numpy as np
import face_recognition
import os
from database import add_face, get_all_faces
import time

class FaceDetector:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.last_face_detection_time = 0
        self.detection_interval = 0.1  # seconds

    def load_known_faces(self):
        """Load known faces from the database."""
        faces = get_all_faces()
        for name, embedding in faces:
            self.known_face_encodings.append(embedding)
            self.known_face_names.append(name)

    def register_new_face(self, frame, name):
        """Register a new face from the current frame."""
        # Find face locations in the frame
        face_locations = face_recognition.face_locations(frame)
        
        if not face_locations:
            return False, "No face detected in the frame"

        # Get face encodings
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        if not face_encodings:
            return False, "Could not generate face encoding"

        # Use the first face encoding
        face_encoding = face_encodings[0]
        
        # Save the face image
        os.makedirs('known_faces', exist_ok=True)
        image_path = f'known_faces/{name}_{int(time.time())}.jpg'
        cv2.imwrite(image_path, frame)

        # Add to database
        if add_face(name, face_encoding, image_path):
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            return True, f"Successfully registered face for {name}"
        else:
            return False, "Failed to save face to database"

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition."""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        current_time = time.time()
        if current_time - self.last_face_detection_time >= self.detection_interval:
            # Find face locations and encodings
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            self.face_names = []
            self.last_face_detection_time = current_time

            # Process each face
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                self.face_names.append(name)

        # Draw results on frame
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        return frame

    def get_face_count(self):
        """Return the number of faces currently detected."""
        return len(self.face_locations) 