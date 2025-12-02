import cv2
from ultralytics import YOLO
import easyocr
import re
import time
import pyttsx3
import numpy as np

# ===============================
# 1. CARGAR MODELO YOLO
# ===============================
print("[INFO] Cargando modelo YOLO...")
modelo_placas = YOLO("placas_ecuador.pt")

# ===============================
# 2. CARGAR EASYOCR
# ===============================
print("[INFO] Cargando OCR EasyOCR...")
lector = easyocr.Reader(['en'], gpu=False)
tts = pyttsx3.init()

# ===============================
# 3. VALIDAR FORMATO DE PLACA ECUATORIANA
# ===============================
def validar_placa_ecuador(placa):
    patron = r"^[A-Z]{3}-?[0-9]{3,4}$"
    return re.match(patron, placa) is not None

def limpiar_texto(txt):
    txt = txt.upper()
    txt = re.sub(r"[^A-Z0-9]", "", txt)
    return txt

def preprocesar_ocr(img):
    if img is None or img.size == 0:
        return img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 9, 75, 75)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    # Morfología para limpiar bordes y unir caracteres impresos
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th

# ===============================
# 4. INICIAR CÁMARA
# ===============================
print("[INFO] Iniciando cámara...")
camara = cv2.VideoCapture(0)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camara.isOpened():
    print("[ERROR] No se pudo acceder a la cámara.")
    exit()

print("\n=== RECONOCIMIENTO DE PLACAS ECUATORIANAS ===")
print("Presiona 'q' para cerrar la ventana o CTRL + C para detener.\n")

# ===============================
# 5. PROCESAMIENTO EN TIEMPO REAL CON VENTANA E INDICACIONES
# ===============================
ultimo_texto = ""
ultimo_tiempo = time.time()

try:
    while True:
        ret, frame = camara.read()
        if not ret:
            print("[ERROR] No se pudo leer la cámara.")
            break

        # Detectar placa con YOLO
        resultados = modelo_placas(frame, verbose=False)

        alguna_lectura = False
        for r in resultados:
            for box in r.boxes:
                try:
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                except Exception:
                    conf = 1.0
                if conf < 0.15:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                placa_crop = frame[y1:y2, x1:x2]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                prep = preprocesar_ocr(placa_crop)
                ocr_result = lector.readtext(prep, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

                if ocr_result:
                    texto = limpiar_texto(ocr_result[0][1])
                    if texto != ultimo_texto and len(texto) >= 5:
                        ultimo_texto = texto
                        ultimo_tiempo = time.time()
                        alguna_lectura = True

                        if validar_placa_ecuador(texto):
                            cv2.putText(frame, f"PLACA: {texto[:3]}-{texto[3:]}", (x1, max(0, y1-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            print(f"[PLACA DETECTADA] → {texto[:3]}-{texto[3:]}")
                            try:
                                tts.say(f"Placa detectada {texto[:3]} {texto[3:]}")
                                tts.runAndWait()
                            except Exception:
                                pass
                        else:
                            cv2.putText(frame, f"TEXTO: {texto}", (x1, max(0, y1-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            print(f"[TEXTO LEÍDO] (No coincide con placa EC): {texto}")

        if not alguna_lectura:
            escalas = [1.0, 1.5, 2.0, 2.5, 3.0]
            for s in escalas:
                h, w = frame.shape[:2]
                fr = cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
                prep_full = preprocesar_ocr(fr)
                ocr_full = lector.readtext(prep_full, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                if ocr_full:
                    texto = limpiar_texto(ocr_full[0][1])
                    if texto != ultimo_texto and len(texto) >= 5:
                        ultimo_texto = texto
                        ultimo_tiempo = time.time()
                        if validar_placa_ecuador(texto):
                            cv2.putText(frame, f"PLACA: {texto[:3]}-{texto[3:]}", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            print(f"[PLACA DETECTADA - GLOBAL] → {texto[:3]}-{texto[3:]}")
                            try:
                                tts.say(f"Placa detectada {texto[:3]} {texto[3:]}")
                                tts.runAndWait()
                            except Exception:
                                pass
                        else:
                            cv2.putText(frame, f"TEXTO: {texto}", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                            print(f"[TEXTO LEÍDO - GLOBAL] (No coincide con placa EC): {texto}")
                        break

        # Mostrar ventana
        cv2.imshow("Camara", frame)

        # Pequeño delay y captura de tecla
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Tecla 'q' presionada. Cerrando...")
            break
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[INFO] Finalizando programa...")

camara.release()
cv2.destroyAllWindows()
print("[INFO] Cámara cerrada.")
