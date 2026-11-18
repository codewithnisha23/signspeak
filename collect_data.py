import argparse
import csv
from pathlib import Path

import cv2

from signspeak.config import DATASET_PATH, DATA_DIR, CAMERA_INDEX
from signspeak.hand_tracker import HandTracker


def main():
    parser = argparse.ArgumentParser(
        description="Collect hand landmark samples for one gesture label."
    )
    parser.add_argument("--label", required=True, help="Gesture label, e.g. HELLO")
    args = parser.parse_args()
    label = args.label.strip().upper().replace(" ", "_")
    if not label:
        raise SystemExit("Label cannot be empty.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not DATASET_PATH.exists()
    tracker = HandTracker()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        tracker.close()
        raise SystemExit("Could not open webcam. Check CAMERA_INDEX in signspeak/config.py.")

    collecting = False
    count = 0
    frame_number = 0
    print("Press S to start/stop collecting. Press Q to quit.")
    try:
        with DATASET_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["label"] + [f"f{i}" for i in range(63)])

            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                features, result = tracker.extract(frame)
                tracker.draw(frame, result)
                frame_number += 1

                if collecting and features is not None and frame_number % 3 == 0:
                    writer.writerow([label] + [float(v) for v in features])
                    count += 1

                state = "COLLECTING" if collecting else "PAUSED"
                cv2.putText(
                    frame, f"{label} | {state} | Samples: {count}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (40, 220, 100) if collecting else (230, 230, 230), 2
                )
                cv2.imshow("SIGNSPEAK – Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    collecting = not collecting
                elif key == ord("q"):
                    break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(f"Saved {count} samples for {label} to {DATASET_PATH}")


if __name__ == "__main__":
    main()
