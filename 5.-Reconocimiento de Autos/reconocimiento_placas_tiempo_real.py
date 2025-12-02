import cv2
from ultralytics import YOLO
import easyocr
import re
import time
import pyttsx3
import numpy as np
import torch
from collections import deque, Counter
import argparse
import os
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0)
parser.add_argument("--conf", type=float, default=0.12)
parser.add_argument("--gpu", type=str, default="auto")
parser.add_argument("--show-fps", action="store_true")
parser.add_argument("--no-tts", action="store_true")
parser.add_argument("--expand", type=float, default=1.25)
parser.add_argument("--weights", type=str, default="placas_ecuador.pt")
parser.add_argument("--proc-skip", type=int, default=2, help="Procesar detección/OCR cada N frames")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

# ===============================
# 1. CARGAR MODELO YOLO
# ===============================
print("[INFO] Cargando modelo YOLO...")
modelo_placas = None
weights_path = args.weights
if os.path.exists(weights_path) and os.path.isfile(weights_path):
    try:
        size_mb = os.path.getsize(weights_path) / (1024*1024)
        print(f"[INFO] Pesos encontrados: {weights_path} ({size_mb:.2f} MB)")
        modelo_placas = YOLO(weights_path)
        print("[INFO] Modelo YOLO cargado correctamente.")
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo cargar el modelo '{weights_path}': {e}")
        print("[ADVERTENCIA] Continuando en modo 'solo OCR' sin detección YOLO.")
else:
    print(f"[ADVERTENCIA] No se encontró el archivo de pesos: {weights_path}")
    print("[ADVERTENCIA] Continuando en modo 'solo OCR' sin detección YOLO.")

# ===============================
# 2. CARGAR EASYOCR
# ===============================
print("[INFO] Cargando OCR EasyOCR...")
auto_cuda = torch.cuda.is_available()
if args.gpu.lower() == "on":
    gpu_flag = True
elif args.gpu.lower() == "off":
    gpu_flag = False
else:
    gpu_flag = bool(auto_cuda)
lector = easyocr.Reader(['en'], gpu=gpu_flag)
tts = None if args.no_tts else pyttsx3.init()

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

def normalizar_placa(txt):
    txt = limpiar_texto(txt)
    if len(txt) < 5:
        return txt
    chars = list(txt)
    for i, c in enumerate(chars):
        if i < 3:
            if c in "012568":
                mapping = {"0":"O","1":"I","2":"Z","5":"S","6":"G","8":"B"}
                chars[i] = mapping.get(c, c)
        else:
            if c in "OISB":
                mapping = {"O":"0","I":"1","S":"5","B":"8"}
                chars[i] = mapping.get(c, c)
    return "".join(chars)

def preprocesar_ocr(img):
    if img is None or img.size == 0:
        return img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    g = cv2.filter2D(g, -1, k)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    # Morfología para limpiar bordes y unir caracteres impresos
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th

def _ordenar_puntos(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, bl, br], dtype="float32")

def rectificar_placa(img):
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3,3), 0)
        e = cv2.Canny(g, 50, 150)
        cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                pts = _ordenar_puntos(pts)
                (tl, tr, bl, br) = pts
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxW = int(max(widthA, widthB))
                maxH = int(max(heightA, heightB))
                if maxW < 30 or maxH < 15:
                    continue
                dst = np.array([[0,0],[maxW-1,0],[0,maxH-1],[maxW-1,maxH-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(pts, dst)
                warp = cv2.warpPerspective(img, M, (maxW, maxH))
                return warp
    except Exception:
        pass
    return img

def ocr_multiescala(img, escalas=(1.0, 1.3, 1.6, 2.0)):
    candidatos = []
    h, w = img.shape[:2]
    for s in escalas:
        rs = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
        prep = preprocesar_ocr(rs)
        res = lector.readtext(prep, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if res:
            txt = normalizar_placa(res[0][1])
            if len(txt) >= 5:
                candidatos.append(txt)
    if not candidatos:
        return ""
    return Counter(candidatos).most_common(1)[0][0]

# ===============================
# 4. INICIAR CÁMARA
# ===============================
print("[INFO] Iniciando cámara...")
try:
    camara = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
except Exception:
    camara = cv2.VideoCapture(args.source)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
try:
    camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

if not camara.isOpened():
    print("[ERROR] No se pudo acceder a la cámara.")
    exit()

print("\n=== RECONOCIMIENTO DE PLACAS ECUATORIANAS ===")
print("Presiona 'q' para cerrar la ventana o CTRL + C para detener.\n")
cv2.namedWindow("Camara", cv2.WINDOW_NORMAL)

# ===============================
# 5. PROCESAMIENTO EN TIEMPO REAL CON VENTANA E INDICACIONES
# ===============================
ultimo_texto = ""
ultimo_tiempo = time.time()
historial_textos = deque(maxlen=15)
ultimo_tts = 0.0
tts_cooldown = 2.5
fps = 0.0
fps_alpha = 0.9
tick_prev = time.time()

def speak_async(texto):
    if tts is None:
        return
    def _run():
        try:
            tts.say(texto)
            tts.runAndWait()
        except Exception:
            pass
    th = threading.Thread(target=_run, daemon=True)
    th.start()

frame_queue = deque(maxlen=1)
result_lock = threading.Lock()
result_data = {"bbox": None, "texto": "", "valido": False}
stop_event = threading.Event()

def worker_loop():
    last_proc = 0
    idx = 0
    while not stop_event.is_set():
        if not frame_queue:
            time.sleep(0.005)
            continue
        if idx % max(1, args.proc_skip) != 0:
            idx += 1
            time.sleep(0.001)
            continue
        frame = frame_queue.popleft()
        idx += 1

        bbox = None
        texto_final = ""
        valido = False

        resultados = []
        if modelo_placas is not None:
            try:
                resultados = modelo_placas(frame, verbose=False)
            except Exception:
                resultados = []

        for r in resultados:
            for box in r.boxes:
                try:
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                except Exception:
                    conf = 1.0
                if conf < args.conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                w = int((x2 - x1) * args.expand)
                h = int((y2 - y1) * args.expand)
                nx1 = max(0, cx - w//2)
                ny1 = max(0, cy - h//2)
                nx2 = min(frame.shape[1]-1, cx + w//2)
                ny2 = min(frame.shape[0]-1, cy + h//2)
                placa_crop = frame[ny1:ny2, nx1:nx2]
                placa_rect = rectificar_placa(placa_crop)
                texto_candidato = ocr_multiescala(placa_rect)
                if len(texto_candidato) >= 5:
                    texto_final = texto_candidato
                    bbox = (nx1, ny1, nx2, ny2)
                    valido = validar_placa_ecuador(texto_final)
                    break
            if texto_final:
                break

        if not texto_final:
            # OCR global espaciado
            if (time.time() - last_proc) > 0.25:
                texto_glob = ocr_multiescala(frame)
                if len(texto_glob) >= 5:
                    texto_final = texto_glob
                    valido = validar_placa_ecuador(texto_final)
                last_proc = time.time()

        with result_lock:
            result_data["bbox"] = bbox
            result_data["texto"] = texto_final
            result_data["valido"] = valido

try:
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()
    last_spoken = ""
    while True:
        ret, frame = camara.read()
        if not ret:
            print("[ERROR] No se pudo leer la cámara.")
            break

        if len(frame_queue) == 0:
            frame_queue.append(frame.copy())

        # Dibujar resultados más recientes
        with result_lock:
            bbox = result_data["bbox"]
            texto = result_data["texto"]
            valido = result_data["valido"]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if texto:
            if valido:
                cv2.putText(frame, f"PLACA: {texto[:3]}-{texto[3:]}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"TEXTO: {texto}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            if texto != last_spoken and tts is not None and (time.time() - ultimo_tts) > tts_cooldown:
                speak_async(f"Placa detectada {texto[:3]} {texto[3:]}")
                ultimo_tts = time.time()
                last_spoken = texto

        # Mostrar ventana
        now = time.time()
        inst_fps = 1.0 / max(1e-3, (now - tick_prev))
        tick_prev = now
        fps = fps_alpha * fps + (1.0 - fps_alpha) * inst_fps
        if args.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Camara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Tecla 'q' presionada. Cerrando...")
            break

except KeyboardInterrupt:
    print("\n[INFO] Finalizando programa...")
finally:
    stop_event.set()
    try:
        worker.join(timeout=2.0)
    except Exception:
        pass

camara.release()
cv2.destroyAllWindows()
print("[INFO] Cámara cerrada.")
