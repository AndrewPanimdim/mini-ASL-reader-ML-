# live_inference.py
import cv2, mediapipe as mp, numpy as np, joblib, json, os

MODEL_DIR = "models"
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_hand_sign.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
    classes = json.load(f)["classes"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def landmarks_to_flat(rel_landmarks):
    return np.array(rel_landmarks).flatten().reshape(1, -1)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]  # first hand
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            # convert to normalized list
            landmarks = [[p.x, p.y, p.z] for p in lm.landmark]
            base = landmarks[0]
            rel = [[x-base[0], y-base[1], z-base[2]] for (x,y,z) in landmarks]
            x = landmarks_to_flat(rel)
            x_s = scaler.transform(x)   # use same scaler
            pred_idx = rf_model.predict(x_s)[0]
            pred_label = classes[pred_idx]
            cv2.putText(frame, f"Pred: {pred_label}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        else:
            cv2.putText(frame, "No hand", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.imshow("Live Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
