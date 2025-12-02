# Reconocimiento de Lenguaje de Señas

Estructura mínima para capturar datos con webcam, extraer keypoints con MediaPipe Hands, entrenar un modelo SVM y hacer inferencia en tiempo real.

## Pasos rápidos

1) Crear y activar entorno (ejemplos Windows PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Recolectar datos (ejemplo para clase "hola"):
```
python src/collect_data.py --label hola --samples 50
```
Repite con más clases cambiando `--label`.

3) Entrenar:
```
python src/train.py
```

4) Inferencia en tiempo real:
```
python src/realtime.py
```

## Estructura
- data/processed/<clase>/*.npy: vectores de keypoints por muestra
- models/model.pkl: modelo SVM
- models/labels.json: mapeo de índices a clases

