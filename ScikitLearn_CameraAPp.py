import base64
import io
import os
from getpass import getpass

import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. OPENROUTER API KEY
# ==========================================
def get_openrouter_key():
    try:
        return getpass("Enter your OpenRouter API key: ").strip()
    except (EOFError, OSError):
        return input("Enter your OpenRouter API key: ").strip()


api_key = os.getenv("OPENROUTER_API_KEY") or get_openrouter_key()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def ask_ai(frame, prompt):
    try:
        small_frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


# ==========================================
# 2. TRAIN A FACE-EMOTION MODEL WITH SCIKIT-LEARN
# ==========================================
# Features: [left_eye_open, right_eye_open, mouth_open, smile_width, brow_tension, face_width]
# Labels: 0=Neutral, 1=Happy, 2=Sad, 3=Angry, 4=Surprised
X_train = np.array(
    [
        [0.22, 0.22, 0.12, 0.20, 0.03, 1.00],  # Neutral
        [0.24, 0.24, 0.35, 0.62, 0.02, 1.00],  # Happy
        [0.18, 0.18, 0.08, 0.16, 0.06, 1.00],  # Sad
        [0.16, 0.16, 0.14, 0.24, 0.08, 1.00],  # Angry
        [0.27, 0.27, 0.42, 0.38, 0.02, 1.00],  # Surprised
    ],
    dtype=float,
)
y_train = np.array([0, 1, 2, 3, 4])

emotion_model = RandomForestClassifier(n_estimators=15, random_state=42)
emotion_model.fit(X_train, y_train)

emotion_names = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]

# ==========================================
# 3. FACE FEATURE EXTRACTION
# ==========================================
def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y]))


def extract_face_features(face_landmarks):
    lm = face_landmarks.landmark

    left_eye_top = lm[159]
    left_eye_bottom = lm[145]
    right_eye_top = lm[386]
    right_eye_bottom = lm[374]

    mouth_top = lm[13]
    mouth_bottom = lm[14]
    mouth_left = lm[61]
    mouth_right = lm[291]

    brow_left = lm[70]
    brow_right = lm[300]

    left_eye_open = distance(left_eye_top, left_eye_bottom)
    right_eye_open = distance(right_eye_top, right_eye_bottom)
    mouth_open = distance(mouth_top, mouth_bottom)
    smile_width = distance(mouth_left, mouth_right)
    brow_tension = abs(distance(brow_left, left_eye_top) - distance(brow_right, right_eye_top))
    face_width = distance(lm[234], lm[454])

    features = np.array(
        [left_eye_open, right_eye_open, mouth_open, smile_width, brow_tension, face_width],
        dtype=float,
    )
    return features.reshape(1, -1)


# ==========================================
# 4. WEBCAM + FACE DETECTION
# ==========================================
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Camera not found or cannot be opened.")

cooldown = 0
ai_text = "Emotion detection ready"
detected_emotion = "No face"
last_prediction = None
stable_count = 0
frame_count = 0

print("Emotion detection started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)

    # Run heavy face detection only every other frame for better speed
    if frame_count % 2 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            mp_draw.draw_landmarks(
                frame,
                face,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            )

            features = extract_face_features(face)
            prediction = int(emotion_model.predict(features)[0])
            current_emotion = emotion_names[prediction]

            if prediction == last_prediction:
                stable_count += 1
            else:
                stable_count = 1
                last_prediction = prediction

            if stable_count >= 3 and cooldown <= 0:
                detected_emotion = current_emotion
                prompt = f"Look at this face and say in one short sentence whether the person seems {detected_emotion.lower()} and why."
                ai_text = ask_ai(frame, prompt)
                cooldown = 30
                stable_count = 0
        else:
            detected_emotion = "No face"
            last_prediction = None
            stable_count = 0

    if cooldown > 0:
        cooldown -= 1

    cv2.putText(
        frame,
        f"Emotion: {detected_emotion}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        ai_text[:80],
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()