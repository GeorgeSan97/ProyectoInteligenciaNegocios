import cv2
import pytesseract
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame
import tempfile
import os
import time
import numpy as np

pygame.mixer.init()

# Si Tesseract no está en PATH, descomenta y corrige la ruta
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocesar_imagen(frame):
    """Preprocesamiento para mejorar reconocimiento manuscrito e impreso."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Suaviza ruido
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Mantiene bordes
    gray = cv2.equalizeHist(gray)  # Mejora contraste global
    # Adaptativa para conservar trazos finos
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    # Invertir a fondo blanco con letras negras
    thresh = cv2.bitwise_not(thresh)
    return thresh

def reconocer_texto(frame):
    """Detecta texto manuscrito o impreso."""
    img_pre = preprocesar_imagen(frame)
    
    # Configuración del motor OCR (más flexible para escritura irregular)
    custom_config = r'--oem 1 --psm 6'
    
    try:
        texto = pytesseract.image_to_string(img_pre, lang='spa+eng', config=custom_config)
    except Exception as e:
        print(f"⚠️ Error OCR: {e}")
        texto = ""
    
    return texto.strip()

def traducir_texto(texto, destino='es'):
    """Traduce automáticamente el texto detectado."""
    if not texto.strip():
        return ""
    try:
        return GoogleTranslator(source='auto', target=destino).translate(texto)
    except Exception as e:
        print(f"⚠️ Error al traducir: {e}")
        return texto

def reproducir_voz(texto):
    """Lee en voz alta el texto traducido."""
    if not texto.strip():
        print("⚠️ No hay texto para reproducir.")
        return
    try:
        tts = gTTS(text=texto, lang='es')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = tmp.name
        tts.save(temp_path)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(temp_path)
    except Exception as e:
        print(f"⚠️ Error al reproducir voz: {e}")

# Inicializa cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara iniciada. Presiona 'S' para reconocer texto o 'Q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo capturar la imagen.")
        break

    cv2.imshow('OCR Manuscrito + Traducción + Voz', frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('s'):
        print("📸 Capturando texto...")
        texto_detectado = reconocer_texto(frame)
        if texto_detectado:
            print("\n🧾 Texto detectado:\n")
            print(texto_detectado)

            traduccion = traducir_texto(texto_detectado, 'es')
            print("\n🌍 Traducción al español:\n")
            print(traduccion)

            print("🔊 Leyendo traducción en voz...\n")
            reproducir_voz(traduccion)
        else:
            print("⚠️ No se detectó texto. Intenta mejorar la iluminación o acercar la cámara.")
    elif tecla == ord('q'):
        print("👋 Saliendo del programa.")
        break

cap.release()
cv2.destroyAllWindows()
