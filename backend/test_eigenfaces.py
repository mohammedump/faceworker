from utils.preprocessing import load_faces_dataset
from sklearn.model_selection import train_test_split
from models.eigenfaces import EigenFacesModel
from sklearn.metrics import accuracy_score
# Charger le dataset
X, y, label_dict = load_faces_dataset("data/faces")
print(f"Dataset chargé : {X.shape} images.")

# Diviser le dataset (80% entrainement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer et entraîner le modèle Eigenfaces
model = EigenFacesModel(n_components=50)
model.train(X_train, y_train)

# Prédiction sur le test
y_pred = model.predict(X_test)

# Calculer et afficher la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Eigenfaces) : {accuracy:.2f}")
model.save("models/eigenfaces_model.pkl")
print("✅ Modèle sauvegardé !")
