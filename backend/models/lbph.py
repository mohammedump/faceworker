import cv2
import numpy as np
import joblib
import os

class LBPHModel:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()

    def train(self, X_train, y_train):
        faces = [img.reshape(100, 100).astype(np.uint8) for img in X_train]
        self.model.train(faces, np.array(y_train))

    def predict(self, X):
        faces = [img.reshape(100, 100).astype(np.uint8) for img in X]
        predictions = []
        for face in faces:
            label, confidence = self.model.predict(face)
            if confidence > 70:  # seuil à ajuster
                predictions.append(-1)
            else:
                predictions.append(label)
        return predictions

    def save(self, path="models/lbph_model.yml"):
        self.model.save(path)

    def load(self, path="models/lbph_model.yml"):
        if os.path.exists(path):
            self.model.read(path)
