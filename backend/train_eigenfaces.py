# backend/train_eigenfaces.py

from utils.preprocessing import load_faces_dataset
from sklearn.model_selection import train_test_split
from models.eigenfaces import EigenFacesModel
from sklearn.metrics import accuracy_score
import os

# Charger les données
X, y, label_dict = load_faces_dataset("backend/data/faces")
print(f"✅ Dataset chargé : {X.shape} images")

# Split entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = EigenFacesModel(n_components=50)
model.train(X_train, y_train)

# Évaluer la précision
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy Eigenfaces : {acc:.2f}")

# Sauvegarder dans models/
os.makedirs("backend/models", exist_ok=True)
model.save("backend/models/eigenfaces_model.pkl")
print("✅ Modèle sauvegardé dans backend/models/eigenfaces_model.pkl")
