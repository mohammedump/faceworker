import sys
import os
import traceback
import pickle
import numpy as np
import cv2
from glob import glob
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition

# Ajout du chemin de base pour les imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.eigenfaces import EigenFacesModel
from models.lbph import LBPHModel
from models.fisherfaces import FisherFacesModel
from utils.preprocessing import load_faces_dataset

app = Flask(__name__)
CORS(app)

# 📁 Détection dynamique du dossier contenant les visages
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/faces"))
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dossier '{dataset_path}' introuvable. Vérifie le chemin.")

# 📦 Chargement des étiquettes
_, _, label_dict = load_faces_dataset(dataset_path)

# 🔐 Chargement des encodages Dlib
try:
    with open("encodings/encodings.pkl", "rb") as f:
        dlib_data = pickle.load(f)
except:
    print("⚠️ Aucun fichier d'encodage Dlib trouvé. Dlib ne pourra pas fonctionner.")
    dlib_data = {"encodings": [], "names": []}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'algo' not in request.form:
        return jsonify({'error': 'Image ou algo manquant'}), 400

    file = request.files['image']
    algo = request.form['algo']
    name = "Inconnu"
    dataset_img = None

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError("❌ Image illisible ou format non supporté.")

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_gray)
        img_resized = cv2.resize(img_eq, (100, 100))

        if algo == "Eigenfaces":
            model_path = os.path.join(os.path.dirname(__file__), "models/eigenfaces_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ Modèle {model_path} manquant.")
            model = EigenFacesModel()
            model.load(model_path)
            pred = model.predict([img_resized.flatten()])[0]
            name = label_dict.get(pred, "Inconnu") if pred != -1 else "Inconnu"

        elif algo == "LBPH":
            model_path = os.path.join(os.path.dirname(__file__), "models/lbph_model.yml")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ Modèle {model_path} manquant.")
            model = LBPHModel()
            model.load(model_path)
            pred = model.predict([img_resized])[0]
            name = label_dict.get(pred, "Inconnu") if pred != -1 else "Inconnu"

        elif algo == "Fisherfaces":
            model_path = os.path.join(os.path.dirname(__file__), "models/fisherfaces_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ Modèle {model_path} manquant.")
            model = FisherFacesModel()
            model.load(model_path)
            pred = model.predict([img_resized.flatten()])[0]
            name = label_dict.get(pred, "Inconnu") if pred != -1 else "Inconnu"

        elif algo == "Dlib":
            rgb_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                results = face_recognition.compare_faces(dlib_data["encodings"], encodings[0])
                if True in results:
                    idx = results.index(True)
                    name = dlib_data["names"][idx]

        if name != "Inconnu":
            match_paths = glob(f"data/faces_png/{name}/*.png")
            if match_paths:
                dataset_img = os.path.basename(match_paths[0])

        return jsonify({"name": name, "dataset_img": dataset_img})

    except Exception as e:
        print("🔥 Erreur pendant la prédiction :")
        traceback.print_exc()
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500

# ✅ Route pour servir les images (nécessaire pour React)
@app.route('/data/faces_png/<name>/<filename>')
def serve_face_image(name, filename):
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/faces_png', name))
    return send_from_directory(folder_path, filename)

# ✅ Route pour retourner la liste des images d'un dossier
@app.route('/list-images/<name>', methods=['GET'])
def list_images(name):
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/faces_png", name))
    if not os.path.exists(folder):
        return jsonify({"error": "Dossier introuvable"}), 404

    images = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    return jsonify({"images": images})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
