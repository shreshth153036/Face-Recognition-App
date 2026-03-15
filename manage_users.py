import os
import shutil
from tkinter import simpledialog, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "app_data", "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "app_data", "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "app_data", "labels.pkl")

def delete_user():
    users = os.listdir(DATASET_DIR)
    if not users:
        messagebox.showinfo("Info", "No users to delete")
        return

    user = simpledialog.askstring(
        "Delete User",
        f"Available users:\n{', '.join(users)}\n\nEnter username:"
    )

    if user and os.path.exists(os.path.join(DATASET_DIR, user)):
        shutil.rmtree(os.path.join(DATASET_DIR, user))
        messagebox.showinfo("Deleted", f"User '{user}' deleted")

def reset_database():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        os.makedirs(DATASET_DIR)

    for f in [MODEL_PATH, LABELS_PATH]:
        if os.path.exists(f):
            os.remove(f)

    messagebox.showinfo("Reset", "Database reset successfully")