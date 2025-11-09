import cv2
import face_recognition
import pickle
import numpy as np

# Cargar modelo entrenado
with open("modelo_entrenado.pkl", "rb") as f:
    encodings_conocidos, nombres_conocidos = pickle.load(f)

# Iniciar cámara (0 = cámara principal)
camara = cv2.VideoCapture(0)

print("🎥 Iniciando detección en tiempo real. Presiona 'q' para salir.")

while True:
    ret, frame = camara.read()
    if not ret:
        print("⚠️ No se pudo acceder a la cámara.")
        break

    # Redimensionar para mayor velocidad
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detectar rostros en la imagen
    ubicaciones = face_recognition.face_locations(rgb_small)
    codificaciones = face_recognition.face_encodings(rgb_small, ubicaciones)

    for (top, right, bottom, left), codificacion in zip(ubicaciones, codificaciones):
        # Comparar con rostros conocidos
        coincidencias = face_recognition.compare_faces(encodings_conocidos, codificacion, tolerance=0.45)
        nombre = "Desconocido"

        if True in coincidencias:
            idx = coincidencias.index(True)
            nombre = nombres_conocidos[idx]

        # Restaurar coordenadas a tamaño original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Definir color según tipo
        if "persona" in nombre:
            color = (0, 255, 0)  # Verde
        elif "animal" in nombre:
            color = (255, 0, 0)  # Azul
        else:
            color = (0, 0, 255)  # Rojo (desconocido)

        # Dibujar recuadro y nombre
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mostrar imagen
    cv2.imshow("Reconocimiento en tiempo real", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
print("🟢 Detección finalizada.")
