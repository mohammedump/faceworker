import face_recognition
import pickle

image_path = "data/faces/s2/1.pgm"  # ← change selon le test
image = face_recognition.load_image_file(image_path)
test_encodings = face_recognition.face_encodings(image)

if not test_encodings:
    print("❌ Aucun visage détecté.")
    exit()

with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)

results = face_recognition.compare_faces(data["encodings"], test_encodings[0])
name = "Inconnu"

if True in results:
    matched_idx = results.index(True)
    name = data["names"][matched_idx]

print(f"✅ Visage reconnu : {name}")
