import threading
import pyttsx3


class SpeechEngine:
    """Runs speech on a background thread so the GUI remains responsive."""

    def speak(self, text):
        cleaned = text.strip()
        if not cleaned:
            return

        def worker():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 155)
                engine.say(cleaned)
                engine.runAndWait()
                engine.stop()
            except Exception as exc:
                print(f"Speech error: {exc}")

        threading.Thread(target=worker, daemon=True).start()
