"""
SIGNSPEAK FastAPI adapter.
Reuses the original signspeak.HandTracker and GestureRecognizer modules.
The browser streams JPEG frames; recognition and model inference stay in Python.
"""
import asyncio, json, time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from signspeak.hand_tracker import HandTracker
from signspeak.recognizer import GestureRecognizer
from signspeak.config import STABLE_FRAMES, GESTURE_COOLDOWN_SECONDS, MODEL_PATH
from signspeak.speech import SpeechEngine

app = FastAPI(title="SIGNSPEAK API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
speech_engine = SpeechEngine()

@app.get("/api/status")
def status():
    recognizer = GestureRecognizer()
    return {
        "ok": True,
        "model_ready": recognizer.ready,
        "model_path": str(MODEL_PATH),
        "message": "Model ready" if recognizer.ready else "Model not trained yet. Collect samples and run train_model.py."
    }

@app.post("/api/speak")
def speak(payload: dict):
    text = str(payload.get("text", "")).strip()
    if text:
        speech_engine.speak(text)
    return {"ok": True, "message": "Speech request accepted"}

@app.websocket("/ws/recognize")
async def recognize(websocket: WebSocket):
    await websocket.accept()
    tracker = HandTracker()
    recognizer = GestureRecognizer()
    last_label: Optional[str] = None
    stable_count = 0
    last_added_at = 0.0
    try:
        await websocket.send_json({
            "type": "ready",
            "model_ready": recognizer.ready,
            "message": "Model ready" if recognizer.ready else "Model not trained yet"
        })
        while True:
            raw = await websocket.receive_bytes()
            np_buffer = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"type":"error","message":"Invalid camera frame"})
                continue
            frame = cv2.flip(frame, 1)
            features, result = tracker.extract(frame)
            landmarks = []
            if result.multi_hand_landmarks:
                for point in result.multi_hand_landmarks[0].landmark:
                    landmarks.append({"x": float(point.x), "y": float(point.y), "z": float(point.z)})
            label, confidence = recognizer.predict(features)
            now = time.time()
            accepted = None
            if label:
                if label == last_label:
                    stable_count += 1
                else:
                    last_label = label
                    stable_count = 1
                if stable_count >= STABLE_FRAMES and now - last_added_at >= GESTURE_COOLDOWN_SECONDS:
                    accepted = label.replace("_", " ")
                    last_added_at = now
                    stable_count = 0
            else:
                last_label = None
                stable_count = 0
            await websocket.send_json({
                "type":"result",
                "hand_detected": bool(landmarks),
                "landmarks": landmarks,
                "label": label,
                "confidence": round(float(confidence), 4),
                "accepted_text": accepted,
                "model_ready": recognizer.ready,
                "message": (
                    f"Gesture recognized: {label}" if accepted else
                    "Hand detected — hold a trained gesture steadily" if landmarks and recognizer.ready else
                    "Show a trained gesture" if recognizer.ready else
                    "Model not trained yet — collect samples and train the model"
                )
            })
    except WebSocketDisconnect:
        pass
    finally:
        tracker.close()
