import os
import cv2
import numpy as np

# Constants
KNOWN_FACES_DIR = 'known_faces'
DATA_DIR = 'data'
DATABASE_PATH = os.path.join(DATA_DIR, 'faces.db')

# Create necessary directories
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def create_blank_image(width=640, height=480):
    """Create a blank image with the specified dimensions."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def save_blank_image():
    """Save a blank image to be used for clearing the frame."""
    blank = create_blank_image()
    cv2.imwrite('blank.jpg', blank)

def get_image_files(directory):
    """Get all image files from the specified directory."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

def format_confidence(confidence):
    """Format confidence value as a percentage string."""
    return f"{confidence * 100:.2f}%"

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=0.6, font_thickness=2, text_color=(255, 255, 255),
                            bg_color=(0, 255, 0)):
    """Draw text with a background rectangle."""
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position
    cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y + 10), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness) 