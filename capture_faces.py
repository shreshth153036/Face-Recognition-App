import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from train_model import train_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "app_data", "dataset")

class FaceRegisterApp:
    def __init__(self, username):
        self.username = username
        self.count = 0

        self.user_dir = os.path.join(DATASET_DIR, username)
        os.makedirs(self.user_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.window = tk.Toplevel()
        self.window.title("Register Face")
        self.window.geometry("640x520")

        self.label = tk.Label(self.window)
        self.label.pack()

        self.counter_label = tk.Label(self.window, text="0 / 50", font=("Arial", 12))
        self.counter_label.pack(pady=5)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(
                os.path.join(self.user_dir, f"{self.count}.jpg"),
                face
            )
            self.count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.counter_label.config(text=f"{self.count} / 50")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.imgtk = ImageTk.PhotoImage(img)
        self.label.config(image=self.imgtk)

        if self.count < 50:
            self.window.after(30, self.update_frame)
        else:
            self.cap.release()
            self.window.destroy()
            train_model()
            messagebox.showinfo(
                "Done",
                f"User '{self.username}' registered and model trained successfully!"
            )

def register_user(username):
    FaceRegisterApp(username)