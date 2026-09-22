import base64
from getpass import getpass
import io
import time
import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. SECURE API KEY INPUT & OPENROUTER CLIENT
# ==========================================
def get_openrouter_key():
    """Prompts for OpenRouter API key using getpass."""
    try:
        return getpass("Enter your OpenRouter API key: ").strip()
    except (EOFError, OSError):
        return input("Enter your OpenRouter API key: ").strip()


api_key = get_openrouter_key()

# OpenRouter acts as an OpenAI-compatible API endpoint
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def ask_openrouter_vision(frame, prompt):
    """Encodes webcam frame to Base64 and sends it to OpenRouter Vision Model."""
    try:
        # Convert OpenCV BGR frame to PIL RGB Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Convert image to Base64 string
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # OpenRouter model endpoint (uses free multimodal router)
        response = client.chat.completions.create(
            model="openrouter/free",  # Automatically uses free vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenRouter API Error: {e}"


# ==========================================
# 2. CREATE & TRAIN SCIKIT-LEARN MODEL
# ==========================================
# Binary finger state vector: [Thumb, Index, Middle, Ring, Pinky]
# Labels: 0 = Other/Fist, 1 = Thumbs Up, 2 = Peace Sign, 3 = Open Palm

X_train = np.array(
    [
        [1, 0, 0, 0, 0],  # Thumbs Up
        [0, 1, 1, 0, 0],  # Peace Sign
        [1, 1, 1, 1, 1],  # Open Palm
        [0, 0, 0, 0, 0],  # Fist
        [0, 1, 0, 0, 0],  # Pointing Finger
    ]
)
y_train = np.array([1, 2, 3, 0, 0])

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# 3. WEBCAM & MEDIAPIPE TRACKING
# ==========================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


def extract_finger_states(landmarks):
    """Extracts 1/0 binary state for 5 fingers from MediaPipe joints."""
    fingers = []

    # Thumb check
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers check
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return np.array(fingers).reshape(1, -1)


# Program State variables
cooldown_counter = 0
ai_response_text = "Perform a gesture to capture webcam photo..."

print("\nWebcam Gesture Assistant Running. Show a gesture to camera. Press 'q' to quit.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_label = "No Gesture"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Predict gesture using Scikit-Learn
            finger_features = extract_finger_states(hand_landmarks.landmark)
            prediction = clf.predict(finger_features)[0]

            if cooldown_counter == 0:
                if prediction in {1, 2, 3}:
                    if prediction == 1:  # Thumbs Up
                        gesture_label = "Thumbs Up Detected!"
                        prompt = "Describe the object or item I am holding in front of the camera in 2 short sentences."
                    elif prediction == 2:  # Peace Sign
                        gesture_label = "Peace Sign Detected!"
                        prompt = "Analyze my face, pose, or expression in this picture and give a short fun comment."
                    elif prediction == 3:  # Open Palm
                        gesture_label = "Open Palm Detected!"
                        prompt = "Summarize everything visible in this webcam photo in detail."

                    ai_response_text = "Snapping webcam photo & asking AI..."
                    cv2.putText(
                        frame,
                        f"Status: {gesture_label}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Webcam Vision Assistant", frame)
                    cv2.waitKey(1)

                    # SNAP PICTURE DIRECTLY FROM WEBCAM FRAME AND SEND TO OPENROUTER
                    ai_response_text = ask_openrouter_vision(frame, prompt)
                    cooldown_counter = 90  # Cooldown clock (~3 seconds)

    if cooldown_counter > 0:
        cooldown_counter -= 1

    # HUD Text Overlays
    cv2.putText(
        frame,
        f"Status: {gesture_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Wrap AI Response text into lines so it fits on screen
    words = ai_response_text.split(" ")
    line1 = " ".join(words[:12])
    line2 = " ".join(words[12:24])

    cv2.putText(
        frame,
        f"AI: {line1}",
        (10, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"    {line2}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Webcam Vision Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()