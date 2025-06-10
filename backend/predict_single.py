import cv2
import numpy as np
from models.eigenfaces import EigenFacesModel
from utils.preprocessing import load_faces_dataset
# Charger le modèle
model = EigenFacesModel()
model.load("models/eigenfaces_model.pkl")

# Charger une image individuelle (change le chemin selon ton dataset)
img = cv2.imread("data/faces/s1/1.pgm", cv2.IMREAD_GRAYSCALE)
img = cv2.equalizeHist(img)  # Si tu as ajouté cette étape dans preprocessing
img_resized = cv2.resize(img, (100, 100)).flatten()

# Prédire
pred = model.predict([img_resized])[0]
print(f"✅ Personne reconnue (ID classe) : {pred}")
# Charger les noms associés aux labels
_, _, label_dict = load_faces_dataset("data/faces")

print(f"✅ Personne reconnue : {label_dict[pred]}")