from utils.preprocessing import load_faces_dataset
from sklearn.model_selection import train_test_split
from models.fisherfaces import FisherFacesModel
from sklearn.metrics import accuracy_score
import time

# Charger dataset
X, y, label_dict = load_faces_dataset("data/faces")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entraîner
n_classes = len(set(y_train))
model = FisherFacesModel(n_components=n_classes - 1)
model.train(X_train, y_train)

# Prédiction + timing
start = time.time()
y_pred = model.predict(X_test)
end = time.time()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Fisherfaces): {accuracy:.2f}")
print(f"Temps de prédiction : {end - start:.4f} s")

# Sauvegarder modèle
model.save()
