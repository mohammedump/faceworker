# realtime_recognition.py
import cv2
import sys
import pickle
import face_recognition

# Lire l'algo
algo = sys.argv[1] if len(sys.argv) > 1 else "eigenfaces"

# Charger encodages Dlib
with open("encodings/encodings.pkl", "rb") as f:
    dlib_data = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Inconnu"
        results = face_recognition.compare_faces(dlib_data["encodings"], face_encoding)
        if True in results:
            idx = results.index(True)
            name = dlib_data["names"][idx]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Reconnaissance Temps Réel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
