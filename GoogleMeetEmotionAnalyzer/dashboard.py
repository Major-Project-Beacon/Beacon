import cv2
import numpy as np
from collections import Counter


class Dashboard:

    def __init__(self):

        self.history = []

    def add_emotion(self, participant_id, emotion):

        self.history.append({
            "participant": participant_id,
            "emotion": emotion
        })

        # Keep memory manageable
        if len(self.history) > 1000:

            self.history.pop(0)

    def get_statistics(self):

        if not self.history:

            return {}

        emotions = [
            item["emotion"]
            for item in self.history
        ]

        counter = Counter(emotions)

        total = len(emotions)

        statistics = {}

        for emotion, count in counter.items():

            statistics[emotion] = round(
                (count / total) * 100,
                1
            )

        return statistics

    def draw(self, frame, participant_emotions, active_participant_ids):

        height, width = frame.shape[:2]

        dashboard_width = 350

        dashboard = np.zeros(
            (height, dashboard_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            dashboard,
            "MEETING ANALYTICS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            dashboard,
            f"Participants: {len(active_participant_ids)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 220, 255),
            2,
        )

        cv2.putText(
            dashboard,
            "PARTICIPANT EMOTIONS",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        y = 145

        for participant_id in sorted(active_participant_ids):
            emotion_data = participant_emotions.get(participant_id, {})
            emotion = str(emotion_data.get("emotion", "Analyzing"))
            confidence = emotion_data.get("confidence")
            confidence_text = (
                f"{float(confidence):.1f}%" if confidence is not None else "--"
            )
            text = f"P{participant_id}: {emotion[:9]:<9} {confidence_text}"

            cv2.putText(
                dashboard,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
            y += 25

        cv2.putText(
            dashboard,
            "EMOTION SUMMARY",
            (20, min(y + 15, height - 100)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        statistics = self.get_statistics()
        y = min(y + 45, height - 65)

        for emotion, percentage in statistics.items():

            text = f"{emotion.upper():<10} {percentage}%"

            cv2.putText(
                dashboard,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y += 40

        separator = np.ones(
            (height, 5, 3),
            dtype=np.uint8
        ) * 255

        output = np.hstack(
            [
                frame,
                separator,
                dashboard
            ]
        )

        return output
        