from deepface import DeepFace
import cv2


class EmotionAnalyzer:

    def __init__(self):
        self.emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral"
        ]

    def analyze(self, face_image):

        try:

            if face_image is None:
                return None

            if face_image.size == 0:
                return None

            result = DeepFace.analyze(
                img_path=face_image,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )

            if isinstance(result, list):
                result = result[0]

            dominant_emotion = result["dominant_emotion"]

            emotion_scores = result["emotion"]

            confidence = emotion_scores[dominant_emotion]

            return {
                "emotion": dominant_emotion,
                "confidence": round(float(confidence), 2),
                "scores": emotion_scores
            }

        except Exception as e:

            print("Emotion analysis error:", e)

            return None