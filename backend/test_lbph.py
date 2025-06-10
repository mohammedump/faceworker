from utils.preprocessing import load_faces_dataset
from sklearn.model_selection import train_test_split
from models.lbph import LBPHModel
from sklearn.metrics import accuracy_score

# Charger les données
X, y, label_dict = load_faces_dataset("data/faces")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entraîner le modèle
model = LBPHModel()
model.train(X_train, y_train)

# Prédire et mesurer la précision
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (LBPH) : {accuracy:.2f}")

# Sauvegarder le modèle
model.save()
