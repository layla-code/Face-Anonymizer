# Video Face Anonymizer
A Python computer vision application designed to detect and anonymize human faces in video files using MediaPipe, OpenCV Haar Cascades, and a memory persistence buffer.

---

## Key Features

- Hybrid Face Detection: Combines Google MediaPipe for frontal faces and OpenCV Haar Cascades for 90-degree side profiles.
- Extended Bounding Margins: Adds a 15 percent margin around detected faces to fully blur ears, chin, and hair lines.
- Anti-Flickering Memory Buffer: Uses Intersection over Union (IoU) tracking to keep faces blurred for up to 15 frames if detection drops temporarily.
- Direct Gaussian Anonymization: Applies Gaussian blur directly onto face regions without drawing bounding boxes.
- Automated Video Pipeline: Resizes frames to 1280x720 resolution, processes video frame-by-frame, and exports the final file.

---

## Technical Methodology

- Environment Setup: Configured Python environment and installed OpenCV with MediaPipe framework.
- Frontal Face Detection: Implemented MediaPipe face detection with custom 15 percent margin scaling.
- Side Profile Detection: Integrated OpenCV profile cascade with frame flipping to detect left and right profiles.
- Overlap Calculation: Developed an IoU algorithm to check overlapping bounding boxes across consecutive frames.
- Persistent Tracking Buffer: Built a 15-frame memory buffer queue to maintain blur during rapid movement.
- Gaussian Blur Application: Extracted face regions of interest and applied a 51x51 kernel Gaussian blur filter.
- Video Processing Pipeline: Configured frame reading, 1280x720 resizing, frame-by-frame preview, and MP4 video export.

---

## Project Directory Structure

Face_detect_blur/
|-- data/
|   |-- test_video.mp4
|   |-- blurred_output.mp4
|-- main.py
|-- README.md

---

## Installation

Install the required libraries by running the following command in your terminal:

pip install opencv-python mediapipe

---

## Usage Instructions

- Step 1: Ensure your input video is placed inside the data directory and named as test_video.mp4.
- Step 2: Execute the main processing script using the following command:

  python main.py
  
- Step 3: View the frame-by-frame window preview and press key q to exit at any time.
- Step 4: Access your processed video saved automatically at data/blurred_output.mp4.

---

## Tech Stack

- Programming Language: Python
- Computer Vision Library: OpenCV
- Detection Framework: Google MediaPipe
