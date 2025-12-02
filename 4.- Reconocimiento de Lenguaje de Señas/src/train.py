from pathlib import Path
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

from config import PROCESSED_DIR, MODELS_DIR


def load_dataset():
    X, y = [], []
    labels = []
    for i, label_dir in enumerate(sorted([d for d in PROCESSED_DIR.glob('*') if d.is_dir()])):
        label = label_dir.name
        labels.append(label)
        for f in label_dir.glob('*.npy'):
            X.append(np.load(f))
            y.append(i)
    if not X:
        raise RuntimeError("No se encontraron datos en data/processed. Recolecta muestras primero.")
    X = np.stack(X)
    y = np.array(y)
    return X, y, labels


def main():
    X, y, labels = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = SVC(kernel='rbf', probability=True, C=10, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, MODELS_DIR / 'model.pkl')
    with open(MODELS_DIR / 'labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print("Modelo guardado en models/model.pkl y labels en models/labels.json")


if __name__ == "__main__":
    main()
