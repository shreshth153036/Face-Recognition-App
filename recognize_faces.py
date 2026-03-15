import cv2
import os
import pickle
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "app_data", "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "app_data", "labels.pkl")

def recognize_faces():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        messagebox.showwarning(
            "No Users",
            "No users registered yet.\nPlease register a user first."
        )
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_count = len(faces)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label, conf = recognizer.predict(face)

            name = label_map.get(label, "Unknown")
            if conf > 75:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (120,200,80), 2)
            cv2.putText(
                frame, f"{name} ({conf:.0f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (120,200,80), 2
            )

        cv2.putText(
            frame,
            f"Face Count: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (225, 105, 65), 2
        )

        cv2.imshow("Face Recognition(Press Q to Stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()