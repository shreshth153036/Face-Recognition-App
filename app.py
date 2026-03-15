import tkinter as tk
from tkinter import simpledialog

from capture_faces import register_user
from recognize_faces import recognize_faces
from manage_users import delete_user, reset_database

root = tk.Tk()
root.title("Face Recognition App")
root.geometry("400x350")
root.resizable(False, False)

main_frame = tk.Frame(root)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

def on_register():
    username = simpledialog.askstring("Register User", "Enter user name:")
    if username:
        register_user(username)  # popup handled inside capture_faces

def on_recognize():
    recognize_faces()

tk.Button(main_frame, text="Register User", width=30, command=on_register).pack(pady=8)
tk.Button(main_frame, text="Start Recognition", width=30, command=on_recognize).pack(pady=8)
tk.Button(main_frame, text="Delete User", width=30, command=delete_user).pack(pady=8)
tk.Button(main_frame, text="Reset Database", width=30, command=reset_database).pack(pady=8)

root.mainloop()