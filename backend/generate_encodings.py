import os
import face_recognition
import pickle

dataset_dir = "backend/data/faces_png"
encodings = []
names = []

for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(person_path, img_name)
            image = face_recognition.load_image_file(img_path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(person)

# Sauvegarde dans encodings.pkl
os.makedirs("encodings", exist_ok=True)
with open("encodings/encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print("✅ Fichier encodings.pkl généré avec succès.")
