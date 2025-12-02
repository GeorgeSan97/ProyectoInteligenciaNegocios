import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from joblib import load

from config import MODELS_DIR

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_hand_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)
    else:
        return np.zeros(63, dtype=np.float32)


essential_files = (MODELS_DIR / 'model.pkl', MODELS_DIR / 'labels.json')


def main(camera: int = 0):
    if not all(p.exists() for p in essential_files):
        print("Faltan archivos de modelo. Entrena primero con src/train.py")
        return

    clf = load(MODELS_DIR / 'model.pkl')
    labels = json.loads((MODELS_DIR / 'labels.json').read_text(encoding='utf-8'))

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            vec = extract_hand_keypoints(results).reshape(1, -1)
            probs = None
            try:
                probs = clf.predict_proba(vec)[0]
                pred_idx = int(np.argmax(probs))
                pred_label = labels[pred_idx]
                pred_conf = float(np.max(probs))
            except Exception:
                pred_idx = int(clf.predict(vec)[0])
                pred_label = labels[pred_idx]
                pred_conf = 1.0

            cv2.putText(image, f"Pred: {pred_label} ({pred_conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Realtime Sign Recognition", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
