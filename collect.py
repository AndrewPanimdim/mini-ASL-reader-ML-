# collect_landmarks.py
import cv2
import mediapipe as mp
import os
import json
import datetime

# -------------------------
# Configuration
# -------------------------
OUT_DIR = "dataset"             # base folder for saved samples
SAVE_IMG = True                 # also save overlay image (helpful for debugging)
MAX_HANDS = 1                   # adjust if you want multi-hand (complexity increases)
MIN_DETECTION_CONF = 0.6
MIN_TRACKING_CONF = 0.6
# -------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_sample_json(label, landmarks, rel_landmarks, out_dir=OUT_DIR, img=None):
    """
    Save one sample to disk as JSON (one file per sample).
    landmarks: list of 21 [x,y,z] normalized coordinates from mediapipe
    rel_landmarks: same shape but relative to wrist (see below)
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label_dir = os.path.join(out_dir, label)
    ensure_dir(label_dir)
    base_name = f"{label}_{ts}"
    json_path = os.path.join(label_dir, base_name + ".json")
    payload = {
        "label": label,
        "timestamp": ts,
        "landmarks": landmarks,            # absolute normalized (x,y,z) returned by mediapipe
        "relative_landmarks": rel_landmarks
    }
    with open(json_path, "w") as f:
        json.dump(payload, f)
    if img is not None and SAVE_IMG:
        img_path = os.path.join(label_dir, base_name + ".jpg")
        cv2.imwrite(img_path, img)
    print(f"[saved] {json_path}")

def landmarks_to_list(hand_landmarks, image_width, image_height):
    """Convert mediapipe normalized landmarks to list of [x,y,z] (normalized x,y in 0..1, z relative)."""
    lm = []
    for p in hand_landmarks.landmark:
        # mediapipe returns x,y normalized (0..1), and z is relative depth
        lm.append([p.x, p.y, p.z])
    return lm

def compute_relative(landmarks):
    """Compute landmarks relative to wrist (index 0). Keeps normalized scale (translation invariant)."""
    base_x, base_y, base_z = landmarks[0]
    rel = [[(x - base_x), (y - base_y), (z - base_z)] for (x,y,z) in landmarks]
    return rel

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")
    print("Webcam opened. Press letter keys (a-z) to save a sample with that label. Press q to quit.")
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)  # <-- mediapipe detection
            annotated = frame.copy()
            if results.multi_hand_landmarks:
                # only first hand (if max_num_hands==1)
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks = landmarks_to_list(hand_landmarks, w, h)      # normalized coords
                rel_landmarks = compute_relative(landmarks)
            else:
                landmarks = None
                rel_landmarks = None

            # overlay instructions
            cv2.putText(annotated, "Press letter key (a-z) to save sample under that label. q to quit.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Collect Hand Landmarks", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # save sample if a letter is pressed and a hand is detected
            if 97 <= key <= 122:  # ascii a..z
                label = chr(key)
                if landmarks is not None:
                    save_sample_json(label, landmarks, rel_landmarks, out_dir=OUT_DIR, img=annotated)
                else:
                    print("No hand detected — sample not saved.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
