import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    STABLE_FRAMES, GESTURE_COOLDOWN_SECONDS
)
from .hand_tracker import HandTracker
from .recognizer import GestureRecognizer
from .speech import SpeechEngine


class SignSpeakApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SIGNSPEAK | Sign Language Recognition")
        self.root.geometry("1000x760")
        self.root.minsize(820, 650)
        self.root.configure(bg="#eef4f1")

        self.tracker = HandTracker()
        self.recognizer = GestureRecognizer()
        self.speech = SpeechEngine()
        self.camera = None
        self.running = False
        self.photo = None
        self.last_label = None
        self.stable_count = 0
        self.last_added_at = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#145c45", padx=20, pady=16)
        header.pack(fill="x")
        tk.Label(
            header, text="SIGNSPEAK", bg="#145c45", fg="white",
            font=("Segoe UI", 23, "bold")
        ).pack(anchor="w")
        tk.Label(
            header, text="Sign Language Recognition and Translation System",
            bg="#145c45", fg="#e0f2e9", font=("Segoe UI", 10)
        ).pack(anchor="w")

        body = tk.Frame(self.root, bg="#eef4f1", padx=18, pady=16)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg="#172b25", bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))
        self.video_label = tk.Label(
            left, text="Camera is stopped", bg="#172b25", fg="white",
            font=("Segoe UI", 14), width=55, height=25
        )
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        right = tk.Frame(body, bg="white", padx=16, pady=16)
        right.pack(side="right", fill="y")
        tk.Label(
            right, text="CONTROL PANEL", bg="white", fg="#145c45",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", pady=(0, 12))

        self.status = tk.StringVar()
        self.status.set(
            "Model ready" if self.recognizer.ready
            else "Model not trained yet"
        )
        tk.Label(
            right, textvariable=self.status, bg="white", fg="#555",
            wraplength=210, justify="left"
        ).pack(anchor="w", pady=(0, 12))

        self._button(right, "Start Camera", self.start_camera, "#16845e")
        self._button(right, "Stop Camera", self.stop_camera, "#66736d")

        tk.Label(
            right, text="TRANSLATED TEXT", bg="white", fg="#145c45",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(22, 7))
        self.text_box = tk.Text(
            right, width=27, height=10, wrap="word",
            font=("Segoe UI", 12), bd=1, relief="solid"
        )
        self.text_box.pack(fill="x")

        self._button(right, "Speak Text", self.speak_text, "#16845e")
        self._button(right, "Clear Text", self.clear_text, "#66736d")

        tk.Label(
            right,
            text="Tip: collect gesture samples and train the model before recognition.",
            bg="white", fg="#666", wraplength=210, justify="left",
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(18, 0))

        footer = tk.Label(
            self.root,
            text="Prototype: recognizes only the static gestures included in your trained dataset.",
            bg="#eef4f1", fg="#53645d", font=("Segoe UI", 9)
        )
        footer.pack(pady=(0, 10))

    def _button(self, parent, text, command, color):
        button = tk.Button(
            parent, text=text, command=command, bg=color, fg="white",
            activebackground=color, activeforeground="white",
            relief="flat", padx=10, pady=9, cursor="hand2",
            font=("Segoe UI", 10, "bold")
        )
        button.pack(fill="x", pady=4)

    def start_camera(self):
        if not self.recognizer.ready:
            messagebox.showwarning(
                "Model not found",
                "First collect samples for at least two gestures, then run:\n"
                "python train_model.py"
            )
            return
        if self.running:
            return
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror(
                "Camera error",
                "Could not open webcam. Close other camera apps or change "
                "CAMERA_INDEX in signspeak/config.py."
            )
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.running = True
        self.status.set("Camera running – show a trained gesture")
        self._update_frame()

    def _update_frame(self):
        if not self.running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.status.set("Unable to read from camera")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        features, result = self.tracker.extract(frame)
        self.tracker.draw(frame, result)

        label, confidence = self.recognizer.predict(features)
        now = time.time()
        if label:
            cv2.putText(
                frame, f"{label} ({confidence:.0%})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 120), 2
            )
            if label == self.last_label:
                self.stable_count += 1
            else:
                self.last_label = label
                self.stable_count = 1

            if (
                self.stable_count >= STABLE_FRAMES
                and now - self.last_added_at >= GESTURE_COOLDOWN_SECONDS
            ):
                self.text_box.insert("end", label.replace("_", " ") + " ")
                self.text_box.see("end")
                self.last_added_at = now
                self.stable_count = 0
        else:
            self.last_label = None
            self.stable_count = 0
            cv2.putText(
                frame, "Show a trained gesture", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((650, 560))
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo, text="")
        self.root.after(30, self._update_frame)

    def stop_camera(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.video_label.configure(image="", text="Camera is stopped")
        self.photo = None
        self.status.set("Camera stopped")

    def speak_text(self):
        self.speech.speak(self.text_box.get("1.0", "end"))

    def clear_text(self):
        self.text_box.delete("1.0", "end")

    def close(self):
        self.stop_camera()
        self.tracker.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
