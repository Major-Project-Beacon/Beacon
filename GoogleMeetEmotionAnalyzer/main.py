import csv
import os
import sys
import time

import cv2
import numpy as np

from dashboard import Dashboard
from emotion_analyzer import EmotionAnalyzer
from face_tracker import FaceTracker
from window_capture import WindowCapture


PROCESS_EVERY_N_FRAMES = 10
OUTPUT_SIZE = (1280, 720)


def open_history_file():
    os.makedirs("data", exist_ok=True)
    path = "data/emotion_history.csv"
    file = open(path, "a", newline="")
    writer = csv.writer(file)
    if os.path.getsize(path) == 0:
        writer.writerow(["timestamp", "participant", "emotion", "confidence"])
    return file, writer


def main():
    print("--------------------------------------")
    print("Google Meet Emotion Analyzer")
    print("--------------------------------------")
    if sys.platform == "darwin":
        print("Grant Screen Recording permission if macOS asks for it.")
    elif sys.platform == "win32":
        print("Keep the Google Meet window visible on your primary monitor.")

    capture = WindowCapture.select()
    print(f"Capturing: {capture.selected_window.label}")
    print("Each detected face receives a participant ID for this session.")
    print("Press Q to quit.")

    emotion_analyzer = EmotionAnalyzer()
    face_tracker = FaceTracker(max_distance=120)
    dashboard = Dashboard()
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    participant_emotions = {}
    frame_count = 0
    csv_file, csv_writer = open_history_file()

    try:
        while True:
            frame = capture.read()
            if frame is None:
                print(
                    "The selected window could not be captured. "
                    "Check System Settings > Privacy & Security > Screen Recording."
                )
                break

            frame = cv2.resize(frame, OUTPUT_SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            boxes = [tuple(box) for box in detected_faces]
            participants = face_tracker.update(boxes)

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                for participant_id, data in participants.items():
                    x, y, width, height = data["box"]
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + width + padding)
                    y2 = min(frame.shape[0], y + height + padding)
                    result = emotion_analyzer.analyze(frame[y1:y2, x1:x2])

                    if result:
                        participant_emotions[participant_id] = {
                            "emotion": result["emotion"],
                            "confidence": result["confidence"],
                        }
                        dashboard.add_emotion(
                            participant_id, result["emotion"]
                        )
                        csv_writer.writerow(
                            [
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                f"Participant {participant_id}",
                                result["emotion"],
                                result["confidence"],
                            ]
                        )
                        csv_file.flush()

            for participant_id, data in participants.items():
                x, y, width, height = data["box"]
                emotion_data = participant_emotions.get(participant_id, {})
                emotion = emotion_data.get("emotion", "Analyzing")
                confidence = emotion_data.get("confidence")
                confidence_text = (
                    f"{float(confidence):.1f}%" if confidence is not None else "--"
                )
                label = f"Participant {participant_id}: {emotion} {confidence_text}"
                label_width = max(260, len(label) * 9)

                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), (0, 255, 0), 2
                )
                label_top = max(0, y - 32)
                cv2.rectangle(
                    frame, (x, label_top), (min(frame.shape[1], x + label_width), y),
                    (0, 0, 0), -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x + 5, max(18, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            cv2.putText(
                frame,
                f"Participants: {len(participants)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            output = dashboard.draw(
                frame, participant_emotions, set(participants.keys())
            )
            cv2.imshow("Google Meet Emotion Analytics", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1
    finally:
        csv_file.close()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nApplication closed.")
    except RuntimeError as exc:
        print(f"Unable to start: {exc}")
