import joblib
import numpy as np

from .config import MODEL_PATH, PREDICTION_CONFIDENCE


class GestureRecognizer:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)

    @property
    def ready(self):
        return self.model is not None

    def predict(self, features):
        if self.model is None or features is None:
            return None, 0.0
        values = np.asarray(features, dtype=np.float32).reshape(1, -1)
        probabilities = self.model.predict_proba(values)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = str(self.model.classes_[best_index])
        if confidence < PREDICTION_CONFIDENCE:
            return None, confidence
        return label, confidence
