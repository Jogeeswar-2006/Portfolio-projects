# AI Security Camera System 🎯

Real-time object detection system with person alert and automatic logging, 
built using YOLOv8 and OpenCV running on GPU.

## Demo
> Run the project and point camera at any scene — 
> detects objects instantly with bounding boxes and labels.

## Features
- Real-time object detection (80 object classes)
- Person detection alert system
- FPS counter display
- Automatic detection logging with timestamps
- GPU accelerated (CUDA - NVIDIA RTX 3050)

## Tech Stack
- Python 3.13
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch (CUDA)
- NVIDIA RTX 3050 GPU

## How to Run

### Install dependencies
pip install ultralytics opencv-python torch torchvision 
--index-url https://download.pytorch.org/whl/cu124

### Run
python detect.py

### Controls
- Press Q to quit
- Detection logs automatically saved to detection_log.txt

## Sample Log Output
2026-05-29 14:32:10 - Persons detected: 1
2026-05-29 14:32:11 - Persons detected: 1
2026-05-29 14:32:15 - Persons detected: 2

## Use Cases
- Security camera systems
- People counting systems
- Edge AI deployment on embedded devices
- Smart surveillance

## Author
Jogeeswar | ECE Student | Embedded Systems & AI# Portfolio Projects - Jogeeswar
