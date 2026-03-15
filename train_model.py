import cv2
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "app_data", "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "app_data", "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "app_data", "labels.pkl")

def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for user in os.listdir(DATASET_DIR):
        user_path = os.path.join(DATASET_DIR, user)
        if not os.path.isdir(user_path):
            continue

        label_map[label_id] = user

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(label_id)

        label_id += 1

    if not faces:
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)