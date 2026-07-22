import os
import cv2
import mediapipe as mp


def is_overlapping(box1, box2):
    """
    Checks if two bounding boxes overlap significantly (Intersection over Union).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area <= 0:
        return False

    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou > 0.2


def process_video_frame(frame, face_detector, profile_cascade, tracked_faces):
    """
    Detects faces, updates face memory buffer (to eliminate flickering),
    and applies Gaussian blur directly to all face regions.
    """
    frame_height, frame_width, _ = frame.shape
    current_detections = []

    # 1. MediaPipe Detection (Frontal / Semi-side faces)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detector.process(rgb_frame)

    if detection_results.detections:
        for face in detection_results.detections:
            bounding_box = face.location_data.relative_bounding_box
            
            # Add margin around face to cover ears and hair
            margin_x = int(bounding_box.width * frame_width * 0.15)
            margin_y = int(bounding_box.height * frame_height * 0.15)

            start_x = max(0, int(bounding_box.xmin * frame_width) - margin_x)
            start_y = max(0, int(bounding_box.ymin * frame_height) - margin_y)
            box_width = int(bounding_box.width * frame_width) + (2 * margin_x)
            box_height = int(bounding_box.height * frame_height) + (2 * margin_y)

            current_detections.append((start_x, start_y, box_width, box_height))

    # 2. OpenCV Haar Cascade Detection (High sensitivity for side profiles)
    if profile_cascade is not None and not profile_cascade.empty():
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Right-facing side profiles
        profiles_right = profile_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.04, minNeighbors=1, minSize=(40, 40)
        )
        for (px, py, pw, ph) in profiles_right:
            current_detections.append((px, py, pw, ph))

        # Left-facing side profiles (Flipped image search)
        flipped_gray = cv2.flip(gray_frame, 1)
        profiles_left = profile_cascade.detectMultiScale(
            flipped_gray, scaleFactor=1.04, minNeighbors=1, minSize=(40, 40)
        )
        for (px, py, pw, ph) in profiles_left:
            original_x = frame_width - px - pw
            current_detections.append((original_x, py, pw, ph))

    # 3. Update Memory Persistence Buffer (Keep faces blurred for 15 frames if briefly lost)
    for det in current_detections:
        matched = False
        for i, tracked in enumerate(tracked_faces):
            tx, ty, tw, th, _ = tracked
            if is_overlapping(det, (tx, ty, tw, th)):
                # Update location and reset timer to 15 frames
                tracked_faces[i] = [det[0], det[1], det[2], det[3], 15]
                matched = True
                break
        if not matched:
            tracked_faces.append([det[0], det[1], det[2], det[3], 15])

    # 4. Apply Blur to all active faces in memory
    updated_tracked = []
    for tracked in tracked_faces:
        x, y, w, h, frames_left = tracked
        
        # Ensure coordinates are inside frame bounds
        start_x = max(0, x)
        start_y = max(0, y)
        end_x = min(frame_width, x + w)
        end_y = min(frame_height, y + h)

        face_region = frame[start_y:end_y, start_x:end_x]
        if face_region.shape[0] > 0 and face_region.shape[1] > 0:
            blurred_face = cv2.GaussianBlur(face_region, (51, 51), 0)
            frame[start_y:end_y, start_x:end_x] = blurred_face

        # Decrease memory timer
        frames_left -= 1
        if frames_left > 0:
            updated_tracked.append([x, y, w, h, frames_left])

    return frame, updated_tracked


# ==========================================
# Main Execution Pipeline
# ==========================================

# Step 1: Define file paths inside the 'data' directory
input_video_path = os.path.join('.', 'data', 'test_video.mp4')
output_video_path = os.path.join('.', 'data', 'blurred_output.mp4')

# Step 2: Open input video file
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {input_video_path}")
else:
    # Step 3: Set display resolution and video writer settings
    target_width = 1280
    target_height = 720
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

    # Step 4: Initialize Detectors
    mp_face_detection = mp.solutions.face_detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    profile_cascade = cv2.CascadeClassifier(cascade_path)

    tracked_faces = []
    print("Processing video and anonymizing all faces with memory buffer... Press 'q' to stop.")

    # Step 5: Processing loop with persistent face memory
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.15) as face_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (target_width, target_height))
            processed_frame, tracked_faces = process_video_frame(
                resized_frame, face_detector, profile_cascade, tracked_faces
            )

            out.write(processed_frame)
            cv2.imshow('Face Anonymizer - Realtime Video', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Step 6: Release resources and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing finished! Saved to: {output_video_path}")