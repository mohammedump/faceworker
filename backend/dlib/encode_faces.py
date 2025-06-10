import face_recognition
import os
import pickle

ENCODINGS_PATH = "encodings/encodings.pkl"
DATASET_DIR = "data/faces/"

known_encodings = []
known_names = []

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person)

os.makedirs("encodings", exist_ok=True)
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"✅ {len(known_names)} visages encodés enregistrés.")
