# Real-Time Deepfake Detection AI

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)

## Description
This project detects **real and deepfake faces** in real-time using a webcam.  
It uses a **Convolutional Neural Network (CNN)** trained on a real vs fake face dataset.  
The system can be further upgraded for video detection or web interface deployment.

## Technologies Used
- Python
- PyTorch
- OpenCV
- TorchVision
- MTCNN (face detection)

## Features
- Real-time webcam face detection
- Classifies Real vs Deepfake faces
- Supports face cropping for more accurate predictions
- Modular code ready for further improvements

## Project Structure

deepfake-detection-ai
-  train_model.py        # Script to train the CNN model for deepfake detection
-  detect_deepfake.py    # Runs real-time webcam detection and classifies Real vs Deepfake
-  dataset_collector.py  # Collects images for building the dataset
-  camera_test.py        # Tests your webcam setup for capturing frames
-  .gitignore            # Ensures large files (dataset, venv, model) are ignored by Git
-  README.md             # This file – explains the project and instructions

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/deepfake-detection-ai.git

2. Navigate into the folder:

cd deepfake-detection-ai

3. Install dependencies:

pip install torch torchvision opencv-python mtcnn matplotlib

## Usage
1. Run real-time detection using your webcam:

python detect_deepfake.py

2. Test your webcam setup:

python camera_test.py

3. Train the deepfake detection model (optional):

python train_model.py

## Future Improvements

- Add deepfake video detection
- Create a web interface using Flask for image upload
- Include heatmap visualization to explain AI predictions
- Train on a larger dataset for higher accuracy