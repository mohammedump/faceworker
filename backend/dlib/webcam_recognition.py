import cv2
import face_recognition
import pickle

# Charger les encodages
with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)

cap = cv2.VideoCapture(0)
print("🎥 Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), encoding in zip(locations, encodings):
        results = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)
        name = "Inconnu"
        if True in results:
            matched_idx = results.index(True)
            name = data["names"][matched_idx]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Reconnaissance - Dlib", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
