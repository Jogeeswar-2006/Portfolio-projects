# ============================================================
# AI Security Camera System
# Built by: Jogeeswar
# Tech: Python, OpenCV, YOLOv8, PyTorch (CUDA - RTX 3050)
# Description: Real-time object detection with person alert
#              system and automatic detection logging
# ============================================================

import cv2                    # OpenCV - handles camera and video
from ultralytics import YOLO  # YOLOv8 - AI detection model
import time                   # Time library - for FPS and timestamps

# -------------------------------------------------------
# LOAD MODEL
# yolov8m.pt = medium model, good accuracy + speed
# Downloads automatically on first run (~50MB)
# -------------------------------------------------------
model = YOLO('yolov8m.pt')

# -------------------------------------------------------
# OPEN WEBCAM
# 0 = default laptop camera
# If you have external camera, change to 1
# -------------------------------------------------------
cap = cv2.VideoCapture(0)

# Variables to calculate FPS
prev_time = 0
curr_time = 0

# Create or open log file to record detections
# 'a' mode = append, so logs keep adding without deleting old ones
log_file = open('detection_log.txt', 'a')

print("AI Security Camera Started...")
print("Press Q to quit")
print("Detection logs saving to detection_log.txt")

# -------------------------------------------------------
# MAIN LOOP
# Runs continuously, processing one frame at a time
# Each frame = one image from webcam
# -------------------------------------------------------
while True:

    # Read one frame from webcam
    # ret = True if frame captured successfully
    # frame = the actual image as a numpy array
    ret, frame = cap.read()

    # If camera disconnected or failed, exit loop
    if not ret:
        print("Camera error. Exiting...")
        break

    # -------------------------------------------------------
    # CALCULATE FPS
    # FPS = Frames Per Second
    # Measures how fast our system processes each frame
    # Formula: FPS = 1 / time taken for one frame
    # -------------------------------------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # -------------------------------------------------------
    # RUN YOLO DETECTION
    # model(frame) runs AI detection on current frame
    # Returns bounding boxes, class labels, confidence scores
    # verbose=False stops cluttering terminal with every detection
    # -------------------------------------------------------
    results = model(frame, verbose=False)

    # -------------------------------------------------------
    # COUNT PERSONS
    # YOLOv8 can detect 80 different object classes
    # Each class has an ID number:
    # 0 = person, 1 = bicycle, 2 = car, 15 = cat, 16 = dog
    # We loop through all detected boxes and count persons
    # -------------------------------------------------------
    person_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # 0 is person
                person_count += 1

    # -------------------------------------------------------
    # DRAW DETECTIONS ON FRAME
    # results[0].plot() draws bounding boxes and labels
    # on top of the original frame automatically
    # -------------------------------------------------------
    annotated = results[0].plot()

    # -------------------------------------------------------
    # DISPLAY FPS ON SCREEN
    # cv2.putText arguments:
    # - frame to write on
    # - text string
    # - position (x, y) from top left
    # - font style
    # - font size
    # - color in BGR format (not RGB)
    # - thickness in pixels
    # -------------------------------------------------------
    cv2.putText(annotated,
                f'FPS: {int(fps)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Display person count on screen in green
    cv2.putText(annotated,
                f'Persons Detected: {person_count}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # -------------------------------------------------------
    # PERSON ALERT SYSTEM
    # If one or more persons detected:
    # 1. Show red alert text on screen
    # 2. Log timestamp and count to detection_log.txt
    # This simulates a real security camera alert system
    # -------------------------------------------------------
    if person_count > 0:

        # Show red alert on screen
        cv2.putText(annotated,
                    'PERSON ALERT',
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # Get current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Write to log file
        log_file.write(f'{timestamp} - Persons detected: {person_count}\n')

        # Flush forces immediate save to file
        # Without this, logs might not save if program crashes
        log_file.flush()

    # -------------------------------------------------------
    # SHOW FINAL FRAME
    # Display annotated frame in a window
    # -------------------------------------------------------
    cv2.imshow('AI Security Camera - Jogeeswar', annotated)

    # -------------------------------------------------------
    # EXIT CONDITION
    # waitKey(1) waits 1ms for a keypress
    # If Q is pressed, break out of loop
    # -------------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------------
# CLEANUP
# Always release camera and close windows properly
# Close log file properly
# -------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
log_file.close()
print("Camera released. Detection logs saved to detection_log.txt")