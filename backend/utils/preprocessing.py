import os
import cv2
import numpy as np

def load_faces_dataset(dataset_path):
    images, labels = [], []
    label_dict = {}

    for i, person in enumerate(sorted(os.listdir(dataset_path))):
        person_path = os.path.join(dataset_path, person)
        label_dict[i] = person
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            images.append(img.flatten())
            labels.append(i)

    return np.array(images), np.array(labels), label_dict