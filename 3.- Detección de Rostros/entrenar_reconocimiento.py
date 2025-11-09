import face_recognition
import cv2
import os
import numpy as np
import pickle

# Rutas
path_personas = "dataset/personas"
path_animales = "dataset/animales"
path_modelo = "modelo_entrenado.pkl"

# Listas
encodings_conocidos = []
nombres_conocidos = []

# Función para procesar carpeta
def procesar_imagenes(path, categoria):
    for nombre_archivo in os.listdir(path):
        if nombre_archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            imagen_path = os.path.join(path, nombre_archivo)
            imagen = face_recognition.load_image_file(imagen_path)
            rostros = face_recognition.face_encodings(imagen)
            if rostros:
                encodings_conocidos.append(rostros[0])
                nombres_conocidos.append(f"{categoria}-{os.path.splitext(nombre_archivo)[0]}")

# Entrenar personas y animales
procesar_imagenes(path_personas, "persona")
procesar_imagenes(path_animales, "animal")

# Guardar modelo
with open(path_modelo, "wb") as f:
    pickle.dump((encodings_conocidos, nombres_conocidos), f)

print("✅ Entrenamiento completado y modelo guardado en:", path_modelo)
