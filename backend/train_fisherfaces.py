import os
import cv2
import numpy as np
from utils.preprocessing import load_faces_dataset
from models.fisherfaces import FisherFacesModel

# Charger les données
dataset_path = os.path.join(os.path.dirname(__file__), "data", "faces_png")
X, y, _ = load_faces_dataset(dataset_path)

# Prétraitement
X_processed = [cv2.resize(cv2.equalizeHist(img), (100, 100)).flatten() for img in X]
X_processed = np.array(X_processed)

# Calcul du max de composantes autorisées
n_classes = len(set(y))
n_features = X_processed.shape[1]
n_components_max = min(n_features, n_classes - 1)

# 👇 Modifier FisherFacesModel pour accepter le paramètre dans train() si nécessaire
model = FisherFacesModel()
model.train(X_processed, y, n_components=n_components_max)

# Sauvegarde
model.save(os.path.join(os.path.dirname(__file__), "models", "fisherfaces_model.pkl"))
print("✅ Modèle Fisherfaces entraîné avec", n_components_max, "composantes LDA.")
