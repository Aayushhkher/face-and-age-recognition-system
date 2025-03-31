# Face and Age Recognition System

Real-time face and age recognition system using OpenCV. Features multi-face detection, age classification (teen/young adult/adult/senior), and face registration. Uses advanced image processing techniques including edge density analysis and texture scoring for accurate age prediction.

## Features

- Real-time face detection and recognition
- Age classification (teen, young adult, adult, senior)
- Multiple face registration support
- Advanced image processing techniques:
  - Edge density analysis
  - Texture scoring
  - Laplacian variance analysis
- User-friendly interface with keyboard controls

## Requirements

- Python 3.8+
- OpenCV (with contrib modules)
- NumPy
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aayushhkher/face-and-age-recognition-system.git
cd face-and-age-recognition-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the face recognition system:
```bash
python advanced_face_recognition.py
```

2. Controls:
- Press 'q' to quit
- Press 'r' to register a new face
- Press 'h' to show help message

## How it Works

The system uses advanced image processing techniques to analyze facial features:
- Edge density analysis for skin texture assessment
- Texture scoring for age-related feature detection
- Laplacian variance analysis for facial detail measurement
- Combined metrics for accurate age classification

## License

This project is licensed under the MIT License - see the LICENSE file for details. 