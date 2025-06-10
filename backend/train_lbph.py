import os
import cv2
import numpy as np
from utils.preprocessing import load_faces_dataset
from models.lbph import LBPHModel

# Charger les images et labels
dataset_path = os.path.join(os.path.dirname(__file__), "data", "faces_png")
X, y, _ = load_faces_dataset(dataset_path)

# Prétraitement : redimensionnement et conversion
X_processed = [cv2.resize(cv2.equalizeHist(img), (100, 100)) for img in X]

# Créer et entraîner le modèle
model = LBPHModel()
model.train(X_processed, y)

# Sauvegarder le modèle
model.save("models/lbph_model.yml")
print("✅ Modèle LBPH entraîné et sauvegardé !")
