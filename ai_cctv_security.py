import cv2
import face_recognition
import numpy as np
import os
import winsound  # Standard library on Windows. Use os.system('say ...') on Mac
from datetime import datetime

# 1. Setup Directories
KNOWN_DIR = "known_faces"
SAVED_DIR = "unknown_visitors"
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(SAVED_DIR, exist_ok=True)

# 2. Load Known Faces
known_encodings = []
known_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_DIR):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        filepath = os.path.join(KNOWN_DIR, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
print(f"Loaded {len(known_names)} known faces.")

# 3. Tapo C210 RTSP Connection
# Replace credentials and IP address with your camera info
RTSP_URL = "rtsp://admin:your_password@192.168.1.100:554/stream2"
cap = cv2.VideoCapture(RTSP_URL)

# Cool-down timer to avoid repeated alerts every second
last_alert_time = 0
ALERT_COOLDOWN = 10  # Seconds between alerts

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Downscale frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare against known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        else:
            # Trigger alert for NEW/UNKNOWN person
            current_time = datetime.now().timestamp()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                last_alert_time = current_time
                
                # A. Save snapshot to local desktop
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(SAVED_DIR, f"unknown_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"[ALERT] Unknown person detected! Saved snapshot to: {image_path}")

                # B. Play Desktop Sound Alert (Frequency: 1000Hz, Duration: 1000ms)
                winsound.Beep(1000, 1000)

        # Draw bounding box around detected face on screen
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4  # Scale back up
        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display camera stream
    cv2.imshow('Tapo C210 - AI Security Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()