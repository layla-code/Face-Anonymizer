# Video Face Anonymizer
A Python computer vision application designed to detect and anonymize human faces in video files using MediaPipe, OpenCV Haar Cascades, and a memory persistence buffer.

## Repository Contents
- Face_detect_blur/
  - data/
    - test_video.mp4: Input video file for processing.
    - blurred_output.mp4: Final exported video with anonymized faces.
  - main.py: Main script handling video processing, detection, and blurring.
  - screenshot.png: Live execution preview showing real-time face blurring.
  - README.md: Project documentation.

## Key Features
- Hybrid Detection: Combines MediaPipe for frontal faces and Haar Cascades for side profiles.
- Extended Margins: Adds a 15% margin to cover ears, chin, and hair lines.
- Anti-Flickering Buffer: Uses IoU tracking and a 15-frame queue for stable tracking during movement.
- Direct Gaussian Blur: Applies a 51x51 kernel blur directly to faces without bounding boxes.
- Automated Pipeline: Resizes frames to 1280x720, displays live preview, and exports MP4 automatically.

## Steps to Build & Recreate the Project
1. Project Setup: Created the directory structure and installed dependencies using pip install opencv-python mediapipe.
2. Data Preparation: Placed the source video inside the data folder and named it test_video.mp4.
3. Script Implementation: Coded main.py to handle frame reading, video scaling, face detection, buffer tracking, and blur application.
4. Program Execution: Ran the script using python main.py, monitored the live processing window, and accessed the final output at data/blurred_output.mp4.

## Tech Stack
- Programming Language: Python
- Computer Vision: OpenCV
- Detection Framework: Google MediaPipe & Haar Cascades
