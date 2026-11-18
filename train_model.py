import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from signspeak.config import DATASET_PATH, MODEL_DIR, MODEL_PATH


def main():
    if not DATASET_PATH.exists():
        raise SystemExit("Dataset not found. Run collect_data.py first.")

    features, labels = [], []
    with DATASET_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        feature_columns = [f"f{i}" for i in range(63)]
        for row in reader:
            try:
                features.append([float(row[column]) for column in feature_columns])
                labels.append(row["label"].strip())
            except (KeyError, ValueError, TypeError):
                continue

    counts = Counter(labels)
    if len(counts) < 2:
        raise SystemExit("Collect samples for at least TWO different gesture labels.")
    if min(counts.values()) < 10:
        raise SystemExit("Each gesture needs at least 10 valid samples; 50–100 is recommended.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    print("Samples per gesture:", dict(counts))

    test_size = max(0.2, len(y) * 0 + 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Test accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
