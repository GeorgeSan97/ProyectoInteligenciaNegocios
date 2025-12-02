import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import mediapipe as mp

from config import PROCESSED_DIR

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_hand_keypoints(results):
    # 21 landmarks x (x, y, z) = 63 features; if no hand, return zeros
    if results.multi_hand_landmarks:
        # pick first detected hand
        hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)
    else:
        return np.zeros(63, dtype=np.float32)


def main(label: str, samples: int, camera: int = 0):
    label_dir = PROCESSED_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    print(f"Recolectando {samples} muestras para la clase '{label}'.")
    print("Presiona 's' para guardar la muestra actual, 'q' para salir.")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        saved = 0
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

            cv2.putText(image, f"Label: {label}  Guardadas: {saved}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Collect Data", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('s') and saved < samples:
                vec = extract_hand_keypoints(results)
                np.save(label_dir / f"{int(time.time()*1000)}.npy", vec)
                saved += 1
                if saved >= samples:
                    print("Colección completa.")
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolectar keypoints de manos con webcam para una clase")
    parser.add_argument("--label", required=True, help="Nombre de la clase")
    parser.add_argument("--samples", type=int, default=50, help="Número de muestras a guardar")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara")
    args = parser.parse_args()
    main(args.label, args.samples, args.camera)
