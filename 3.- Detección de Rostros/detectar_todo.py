import face_recognition
import cv2
import pickle
import numpy as np
import os

# Cargar modelo entrenado
with open("modelo_entrenado.pkl", "rb") as f:
    encodings_conocidos, nombres_conocidos = pickle.load(f)

# Carpeta con imágenes de prueba
carpeta_pruebas = "pruebas"

for archivo in os.listdir(carpeta_pruebas):
    if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path_imagen = os.path.join(carpeta_pruebas, archivo)
    imagen = cv2.imread(path_imagen)

    # Convertir a RGB
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Detectar rostros
    ubicaciones = face_recognition.face_locations(rgb)
    codificaciones = face_recognition.face_encodings(rgb, ubicaciones)

    for (top, right, bottom, left), codificacion in zip(ubicaciones, codificaciones):
        coincidencias = face_recognition.compare_faces(encodings_conocidos, codificacion)
        nombre = "Desconocido"

        if True in coincidencias:
            idx = coincidencias.index(True)
            nombre = nombres_conocidos[idx]

        # Dibujar rectángulo y nombre
        color = (0, 255, 0) if "persona" in nombre else (255, 0, 0)
        cv2.rectangle(imagen, (left, top), (right, bottom), color, 2)
        cv2.putText(imagen, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mostrar resultado
    cv2.imshow("Detección", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
